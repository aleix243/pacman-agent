# Licensing Information: You are free to use or extend these projects for
# educational purposes provided that you do not distribute or publish solutions,
# retain this notice, and attribute UC Berkeley (http://ai.berkeley.edu).
# Pacman AI projects: John DeNero, Dan Klein and the CS188 contributors.
"""Capture team: risk-weighted graph search, sonar tracking and adaptive roles.
Only uses observations supplied by the contest framework; no external data.
"""
from collections import Counter, deque
from heapq import heappop, heappush
from functools import lru_cache
from contest.capture_agents import CaptureAgent
from contest.game import Directions

STEPS = ((Directions.NORTH, 0, 1), (Directions.SOUTH, 0, -1),
         (Directions.EAST, 1, 0), (Directions.WEST, -1, 0))


def create_team(first_index, second_index, is_red, **kwargs):
    return [StrategicAgent(first_index, True), StrategicAgent(second_index, False)]


def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


class StrategicAgent(CaptureAgent):
    def __init__(self, index, primary):
        super().__init__(index)
        self.primary = primary

    def register_initial_state(self, state):
        super().register_initial_state(state)
        self.start = state.get_agent_position(self.index)
        self.mate = next(i for i in self.get_team(state) if i != self.index)
        self.enemies = self.get_opponents(state)
        walls = state.get_walls()
        self.mid = walls.width // 2
        self.nodes = set(walls.as_list(False))
        self.home = {p for p in self.nodes if (p[0] < self.mid) == self.red}
        self.away = self.nodes - self.home
        self.graph = {p: [(a, (p[0]+dx, p[1]+dy)) for a,dx,dy in STEPS
                          if (p[0]+dx,p[1]+dy) in self.nodes] for p in self.nodes}
        self.gates = sorted(p for p in self.home if any(q in self.away for _,q in self.graph[p]))
        self.dist = self.distancer._distances
        self.home_dist = {p: min((self.d(p,g) for g in self.gates), default=999) if p in self.away else 0 for p in self.nodes}
        # Peel dead-end branches, recording the junction each branch must exit.
        degree = {p:len(self.graph[p]) for p in self.away}
        queue = deque(p for p in degree if degree[p] <= 1)
        parent = {}
        while queue:
            p = queue.popleft()
            if p not in degree or degree[p] > 1:
                continue
            neighbours = [q for _,q in self.graph[p] if q in degree or q in self.home]
            del degree[p]
            if neighbours:
                q = neighbours[0]
                parent[p] = q
                if q in degree:
                    degree[q] -= 1
                    if degree[q] <= 1: queue.append(q)
        self.trap = {}
        for p in parent:
            q = p
            while q in parent:
                q = parent[q]
            if q != p: self.trap[p] = (q, self.d(p,q))
        self.belief = {i:{state.get_initial_agent_position(i):1.0} for i in self.enemies}
        self.last_seen = {}
        self.prev_food = set(self.get_food_you_are_defending(state).as_list())
        self.prev_caps = set(self.get_capsules_you_are_defending(state))
        self.evidence = []
        self.recent = deque(maxlen=12)
        self.target = None
        self.mode = None
        self.returning = False
        self.last_pos = None
        self.last_time = None
        self.exit_dist = {p:min((self.d(p,g) for g in self.gates),default=999)+1 for p in self.home}
        self.home_moves = {p:tuple(q for _,q in self.graph[p] if q in self.home)+(p,) for p in self.home}

    def d(self, a, b):
        return self.dist.get((a,b), 10000)

    def track(self, state, pos, mate, now):
        food = set(self.get_food_you_are_defending(state).as_list())
        caps = set(self.get_capsules_you_are_defending(state))
        missing = (self.prev_food-food) | (self.prev_caps-caps)
        self.prev_food, self.prev_caps = food,caps
        self.evidence = [(p,t) for p,t in self.evidence if t-now <= 40]
        self.evidence.extend((p,now) for p in sorted(missing))
        noisy = state.get_agent_distances()
        for i in self.enemies:
            e = state.get_agent_state(i)
            seen = state.get_agent_position(i)
            if seen is not None:
                prev = self.last_seen.get(i)
                self.last_seen[i] = (seen, now, prev[0] if prev and prev[1]-now <= 4 else None)
                self.belief[i] = {seen:1.0}
                continue
            side = self.home if e.is_pacman else self.away
            possible = {p for p in side if manhattan(p,pos)>5 and manhattan(p,mate)>5
                        and (not noisy or state.get_distance_prob(manhattan(p,pos),noisy[i])>0)}
            if not possible:
                self.belief[i] = {}
                continue
            prior = Counter()
            for p,w in self.belief[i].items():
                adj = [q for _,q in self.graph.get(p,())] + [p]
                for q in adj:
                    if q in possible: prior[q] += w/len(adj)
            # Keep a small recovery mass: a hidden enemy may have respawned.
            floor = .025 / len(possible)
            weights = {p:prior[p]+floor for p in possible}
            if e.is_pacman and missing:
                for p in weights:
                    dist = min(self.d(p,q) for q in missing)
                    weights[p] *= 8 if dist==0 else 4 if dist==1 else 1
            total = sum(weights.values())
            self.belief[i] = {p:w/total for p,w in weights.items()}

    def search(self, pos, costs, legal, home_only=False):
        """One weighted Dijkstra pass: cost and first legal action to every tile."""
        dist, first = {pos:0.0}, {}
        heap = [(0.0,pos)]
        allowed = set(legal)
        while heap:
            value,p = heappop(heap)
            if value != dist[p]: continue
            for a,q in self.graph[p]:
                if p == pos and a not in allowed: continue
                if home_only and q not in self.home: continue
                nv = value + costs.get(q,1.0)
                if nv < dist.get(q,1e12):
                    dist[q] = nv
                    first[q] = a if p == pos else first[p]
                    heappush(heap,(nv,q))
        return dist,first

    def threat(self,state):
        visible, hidden = [], []
        for i in self.enemies:
            e=state.get_agent_state(i)
            p=state.get_agent_position(i)
            if not e.is_pacman:
                if p is not None: visible.append((p,e.scared_timer))
                elif e.scared_timer <= 5:
                    hidden.extend(sorted(self.belief[i].items(),key=lambda kv:-kv[1])[:8])
        return visible,hidden

    def risks(self,state,pos,caps,visible,hidden,carrying):
        counts=Counter(self.recent)
        costs={}
        for p in self.nodes:
            c=1+.22*counts[p]
            if p in self.away and p not in caps:
                for g,t in visible:
                    d=self.d(p,g)
                    arrival=self.d(pos,p)
                    if t>arrival+2: continue
                    # Immediate capture risk dominates; further risk is soft to
                    # permit alternate routes around guards rather than freezing.
                    c += (65+6*carrying if d<=1 else 12 if d==2 else 4 if d==3 else 1.2 if d==4 else 0)
                    if p in self.trap:
                        exit_p,depth=self.trap[p]
                        if self.d(g,exit_p)+t <= depth+2 and d<=depth+5:
                            c+= min(15,2*depth)
                c += sum(w*(2.5 if self.d(p,g)<=2 else .5 if self.d(p,g)<=4 else 0) for g,w in hidden)
            costs[p]=c
        return costs

    def tactical_legal(self,state,pos,legal,me):
        """Filter moves that allow a visible opponent to capture us next turn.
        Capsules take effect before collision; crossing home deposits food.
        This is a conservative one-round check, not a prediction of intent.
        """
        caps=set(self.get_capsules(state))
        moves={a:q for a,q in self.graph[pos] if a in legal}
        if Directions.STOP in legal: moves[Directions.STOP]=pos
        hazards={}
        for a,q in moves.items():
            risk=0
            for i in self.enemies:
                e=state.get_agent_state(i); g=state.get_agent_position(i)
                if g is None: continue
                if q in self.away and q not in caps:
                    if e.scared_timer==0 and g==q: risk+=2
                    elif e.scared_timer==0 and any(z==q for _,z in self.graph[g]): risk+=1
                elif q in self.home and me.scared_timer>0:
                    if g==q: risk+=2
                    elif me.scared_timer>1 and any(z==q for _,z in self.graph[g]): risk+=1
            hazards[a]=risk
        if not hazards: return legal
        safe=[a for a,r in hazards.items() if r==0]
        # Allow a safe stop only if all movement alternatives are dangerous.
        if safe: return safe
        return [a for a,r in hazards.items() if r==min(hazards.values())]

    def forced_capture(self,pos,legal,q):
        """Three defender turns, adversarial replies, no hidden-state access.
        Return a guaranteed capture if one exists; otherwise defer to strategy.
        """
        @lru_cache(maxsize=None)
        def solve(p,e,depth):
            if p==e: return 0
            if depth==0 or self.d(p,e)>depth*2: return 99
            best=99
            for z in self.home_moves[p]:
                if z==e: return 1
                replies=[v for _,v in self.graph[e] if v!=z]+[e]
                if any(v in self.away for v in replies): continue
                worst=max((solve(z,v,depth-1) for v in replies),default=0)
                best=min(best,1+worst)
            return best
        options=[]
        for a,z in self.graph[pos]+[(Directions.STOP,pos)]:
            if a not in legal or z not in self.home: continue
            if z==q: return a
            replies=[v for _,v in self.graph[q] if v!=z]+[q]
            if any(v in self.away for v in replies): continue
            worst=max((solve(z,v,2) for v in replies),default=0)
            if worst<90: options.append((worst,a))
        return min(options)[1] if options else None

    def choose_action(self,state):
        legal=state.get_legal_actions(self.index)
        if not legal: return Directions.STOP
        me=state.get_agent_state(self.index)
        other=state.get_agent_state(self.mate)
        pos=state.get_agent_position(self.index)
        mate=state.get_agent_position(self.mate)
        now=state.data.timeleft
        if self.last_pos is not None and self.d(self.last_pos,pos)>1:
            self.recent.clear(); self.returning=False; self.target=None
        self.last_pos=pos
        self.recent.append(pos)
        self.track(state,pos,mate,now)
        legal=self.tactical_legal(state,pos,legal,me)
        # A visible invader one step away can be captured now, even when
        # the strategic assignment would otherwise make this agent attack.
        if not me.is_pacman and me.scared_timer==0:
            for i in self.enemies:
                q=state.get_agent_position(i)
                if q in self.home and state.get_agent_state(i).is_pacman:
                    for a,z in self.graph[pos]:
                        if a in legal and z==q: return a
        invaders=[i for i in self.enemies if state.get_agent_state(i).is_pacman]
        score=self.get_score(state)
        turns=(now+3)//4
        # Assign one defender when needed, favour an agent already at home.
        defend=False
        if invaders:
            def responsibility(p,agent):
                vals=[]
                for i in invaders:
                    v=state.get_agent_position(i)
                    if v is not None: vals.append(self.d(p,v))
                    elif self.belief[i]: vals.append(sum(w*self.d(p,q) for q,w in self.belief[i].items()))
                return (min(vals,default=10)+ (6 if agent.is_pacman else 0)
                        + 1.8*agent.num_carrying + min(12,agent.scared_timer)*.7)
            ours=responsibility(pos,me); theirs=responsibility(mate,other)
            defend=ours<theirs or (ours==theirs and not self.primary)
            # A large late lead justifies both defending if possible.
            if score>2 and turns<50 and not me.is_pacman: defend=True
        elif score>0 and turns<35:
            defend=not self.primary or score>=5
        elif score>=6 and not self.primary and not me.is_pacman:
            defend=True
        # Assign separate visible invaders when protecting a lead.
        if (len(invaders)==2 and score>0 and not me.is_pacman
                and not other.is_pacman and me.scared_timer==0
                and other.scared_timer==0):
            ordered=sorted(invaders)
            a,b=(state.get_agent_position(i) for i in ordered)
            if a in self.home and b in self.home:
                own=self.d(pos,a)+self.d(mate,b)
                swap=self.d(pos,b)+self.d(mate,a)
                invaders=[ordered[0] if own<swap or (own==swap and self.primary) else ordered[1]]
                defend=True
        if defend and me.scared_timer>8 and score<=0 and not me.num_carrying:
            defend=False
        mode='defend' if defend else 'attack'
        if mode!=self.mode:
            self.target=None
            self.mode=mode
        if defend and not me.is_pacman:
            self.returning=False
            return self.defend(state,pos,legal,invaders,me)
        return self.attack(state,pos,mate,legal,me,turns,score,defend)

    def attack(self,state,pos,mate,legal,me,turns,score,recall):
        foods=set(self.get_food(state).as_list())
        caps=set(self.get_capsules(state))
        carrying=me.num_carrying
        visible,hidden=self.threat(state)
        costs=self.risks(state,pos,caps,visible,hidden,carrying)
        # A scared ghost on our own territory is vulnerable to enemy Pacmen.
        if me.scared_timer>0:
            for i in self.enemies:
                e=state.get_agent_state(i); q=state.get_agent_position(i)
                if e.is_pacman and q is not None:
                    for p in self.home:
                        if self.d(p,q)<=1: costs[p]+=50
        dist,first=self.search(pos,costs,legal)
        gates=[g for g in self.gates if g in first]
        safe_gate=min(gates,key=lambda g:dist[g],default=None)
        danger=min((self.d(pos,g) for g,t in visible if t<=4),default=100)
        # Harvest during the remaining capsule window, with escape margin.
        shield=min(state.get_agent_state(i).scared_timer for i in self.enemies)
        protected=shield>self.home_dist[pos]+6
        if not me.is_pacman: self.returning=False
        if me.is_pacman:
            threshold=7 if danger>=7 else 4 if danger>=4 else 2
            if score>0: threshold=max(2,threshold-1)
            if score< -5 and danger>=7: threshold=9
            if protected and turns>self.home_dist[pos]+8 and not recall:
                threshold=max(threshold, 11 if score<=0 else 9)
            if (carrying>=threshold or (carrying and turns<=self.home_dist[pos]+5)
                    or (carrying and len(foods)<=2) or recall): self.returning=True
            if carrying>=1 and danger<=2: self.returning=True
        if self.returning and safe_gate is not None:
            # A capsule can make an otherwise blocked retreat safe.
            usable=[c for c in caps if c in first and self.d(pos,c)<=3 and dist[c]<dist[safe_gate]*.65 and danger<5]
            target=min(usable,key=lambda p:dist[p],default=safe_gate)
            self.target=target
            return first[target]
        # Select a food cluster by travel cost, density, escape access and
        # separation from our teammate. Replan every turn, keep a mild target bonus.
        candidates=[]
        mate_attacking=mate in self.away or not self.primary
        for f in foods:
            if f not in first: continue
            near=sum(1 for q in foods if q!=f and self.d(f,q)<=3)
            value=dist[f]+.24*self.home_dist[f]-.85*min(near,5)
            if mate_attacking and self.d(mate,f)+2<self.d(pos,f): value+=4.5
            if f==self.target: value-=1.1
            if f in self.trap and visible:
                exit_p,depth=self.trap[f]
                for g,t in visible:
                    if self.d(g,exit_p)+t <= self.d(pos,f)+depth+2:
                        value+=min(16,2.0*depth)
            if self.home_dist[f]+self.d(pos,f)+2>turns: value+=40
            candidates.append((value,f))
        for c in caps:
            if c in first:
                active=any(t<=self.d(pos,c)+6 for _,t in visible)
                # Preserve a second capsule if protection is still ample.
                fresh=shield>self.d(pos,c)+10
                bonus=-3 if fresh else (12 if active else 5)
                candidates.append((dist[c]+.15*self.home_dist[c]-bonus,c))
        if candidates:
            _,target=min(candidates)
            self.target=target
            return first[target]
        if me.is_pacman and safe_gate is not None: return first[safe_gate]
        return self.fallback(pos,legal,costs)

    def defend(self,state,pos,legal,invaders,me):
        costs={p:1.0 for p in self.home}
        if me.scared_timer>0:
            for i in invaders:
                q=state.get_agent_position(i)
                if q is not None:
                    for p in self.home:
                        d=self.d(p,q)
                        costs[p]+=80 if d<=1 else 12 if d==2 else 0
        if me.scared_timer==0:
            nearby=sorted((self.d(pos,q),i,q) for i in invaders
                          for q in [state.get_agent_position(i)] if q in self.home)
            for d,i,q in nearby:
                if d<=5:
                    a=self.forced_capture(pos,legal,q)
                    if a is not None: return a
        dist,first=self.search(pos,costs,legal,home_only=True)
        targets=[]
        for i in invaders:
            q=state.get_agent_position(i)
            e=state.get_agent_state(i)
            if q is not None:
                if me.scared_timer==0:
                    # Chase directly if close; otherwise cut off an exit when
                    # carrying enemies are likely to retreat.
                    if q in dist:
                        targets.append((dist[q]-4-.5*e.num_carrying,q))
                    if self.d(pos,q)>2 and e.num_carrying>=2:
                        # Candidate interception tiles must lie near a short
                        # exit route. Avoid holding a random far-off gate.
                        exits=sorted(self.gates,key=lambda g:self.d(q,g))
                        if exits:
                            best_exit=self.d(q,exits[0])
                            likely=[g for g in exits if self.d(q,g)<=best_exit+2]
                            previous=self.last_seen.get(i,(None,None,None))[2]
                            for g in likely:
                                if g not in dist or dist[g]>self.d(q,g): continue
                                heading=0 if previous is None else self.d(previous,g)-self.d(q,g)
                                targets.append((dist[g]+.5*self.d(q,g)-4-.8*e.num_carrying-1.5*heading,g))
                else:
                    for p in self.home:
                        if p in dist and self.d(p,q)==3:
                            targets.append((dist[p]-3,p))
            else:
                belief=self.belief[i]
                for p,w in sorted(belief.items(),key=lambda kv:-kv[1])[:12]:
                    if p in dist: targets.append((dist[p]-12*w-2,p))
        if not targets and self.evidence and invaders:
            for p,t in self.evidence:
                if p in dist: targets.append((dist[p]+.15*(t-state.data.timeleft),p))
        if not targets:
            food=self.prev_food
            for g in self.gates:
                if g in dist:
                    coverage=sum(1 for f in food if self.d(g,f)<=6)
                    mate=state.get_agent_position(self.mate)
                    overlap=4 if self.d(mate,g)+2<self.d(pos,g) else 0
                    targets.append((.5*dist[g]-.8*coverage+overlap,g))
        if targets:
            _,target=min(targets)
            self.target=target
            if target in first: return first[target]
            # Holding an interception point is preferable to stepping off it.
            if Directions.STOP in legal: return Directions.STOP
        return self.fallback(pos,legal,costs,True)

    def fallback(self,pos,legal,costs,home_only=False):
        choices=[(costs.get(q,100)+.5*self.recent.count(q),a) for a,q in self.graph[pos]
                 if a in legal and (not home_only or q in self.home)]
        return min(choices)[1] if choices else (Directions.STOP if Directions.STOP in legal else legal[0])
