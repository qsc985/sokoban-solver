// Sokoban solver — push-optimal per room.
//
// The generator's levels are a hallway (one row, where the player starts) with
// 1-4 independent rooms off it (3-4 boxes each, one door each). Boxes never
// leave their room in an optimal solution, so a level's push-optimum is the sum
// of its rooms' optima. This solver detects the hallway and each room, solves
// every room push-optimally with A* over pushes, and stitches the room
// solutions together with hallway walks.
//
// Per room, A* runs over (player, sorted boxes). A successor is one push: the
// player walks (BFS) to the square behind a box and pushes it one step. Cost is
// 1 per push. The heuristic is the min-cost assignment (Hungarian, brute-forced
// for <=4 boxes) of boxes to goals using precomputed BFS distances — admissible,
// so A* returns the push-optimum. Dead states are pruned with a reverse-push
// live-cell analysis (a box that cannot reach any goal even ignoring other boxes
// and the player is truly dead) and a 2x2 freeze check. Everything is
// deterministic: fixed direction order, FIFO tie-breaking, work-based caps only
// (no clock, no randomness), so two runs produce byte-identical output.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

interface Dir {
  dx: number;
  dy: number;
  ch: string;
}
const DIRS: Dir[] = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
];

interface Level {
  w: number;
  h: number;
  wall: boolean[];
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall: boolean[] = new Array(w * h).fill(true);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      wall[i] = ch === '#';
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// ---------- BFS helpers ----------

function bfsFrom(lv: Level, from: number, blocked: Set<number>): { prev: Int32Array; prevDir: string[] } {
  const prev = new Int32Array(lv.w * lv.h).fill(-1);
  const prevDir: string[] = new Array(lv.w * lv.h).fill('');
  prev[from] = from;
  const q: number[] = [from];
  let head = 0;
  while (head < q.length) {
    const c = q[head] as number;
    head++;
    const cx = c % lv.w;
    const cy = Math.floor(c / lv.w);
    for (const d of DIRS) {
      const nx = cx + d.dx;
      const ny = cy + d.dy;
      if (nx < 0 || ny < 0 || nx >= lv.w || ny >= lv.h) continue;
      const n = ny * lv.w + nx;
      if (lv.wall[n] || blocked.has(n) || prev[n] !== -1) continue;
      prev[n] = c;
      prevDir[n] = d.ch;
      q.push(n);
    }
  }
  return { prev, prevDir };
}

function pathTo(prev: Int32Array, prevDir: string[], from: number, to: number): string | null {
  if (prev[to] === -1) return null;
  let s = '';
  let c = to;
  while (c !== from) {
    s = prevDir[c] + s;
    c = prev[c];
  }
  return s;
}

// BFS distances from one source over a set of floor cells (ignoring boxes).
function bfsDist(lv: Level, from: number, allowed: Set<number>): number[] {
  const dist = new Array<number>(lv.w * lv.h).fill(Infinity);
  dist[from] = 0;
  const q: number[] = [from];
  let head = 0;
  while (head < q.length) {
    const c = q[head] as number;
    head++;
    const cx = c % lv.w;
    const cy = Math.floor(c / lv.w);
    for (const d of DIRS) {
      const nx = cx + d.dx;
      const ny = cy + d.dy;
      if (nx < 0 || ny < 0 || nx >= lv.w || ny >= lv.h) continue;
      const n = ny * lv.w + nx;
      if (!allowed.has(n) || dist[n] !== Infinity) continue;
      dist[n] = dist[c] + 1;
      q.push(n);
    }
  }
  return dist;
}

// ---------- room detection ----------

interface Room {
  cells: number[];
  cellSet: Set<number>;
  boxes: number[];
  goals: number[];
  door: number;
}

function findRooms(lv: Level): Room[] {
  const hy = Math.floor(lv.player / lv.w);
  const seen = new Array<boolean>(lv.w * lv.h).fill(false);
  const rooms: Room[] = [];
  for (let y = 0; y < lv.h; y++) {
    for (let x = 0; x < lv.w; x++) {
      const i = y * lv.w + x;
      if (y === hy || lv.wall[i] || seen[i]) continue;
      const cells: number[] = [];
      const stack: number[] = [i];
      seen[i] = true;
      while (stack.length > 0) {
        const c = stack.pop() as number;
        cells.push(c);
        const cx = c % lv.w;
        const cy = Math.floor(c / lv.w);
        for (const d of DIRS) {
          const nx = cx + d.dx;
          const ny = cy + d.dy;
          if (ny === hy || nx < 0 || ny < 0 || nx >= lv.w || ny >= lv.h) continue;
          const ni = ny * lv.w + nx;
          if (lv.wall[ni] || seen[ni]) continue;
          seen[ni] = true;
          stack.push(ni);
        }
      }
      const cellSet = new Set(cells);
      const boxes = lv.boxes.filter((b) => cellSet.has(b));
      const goals = lv.goals.filter((g) => cellSet.has(g));
      if (boxes.length === 0) continue;
      const door = cells.find((c) => {
        const cy = Math.floor(c / lv.w);
        return cy === hy - 1 || cy === hy + 1;
      }) ?? cells[0] as number;
      rooms.push({ cells, cellSet, boxes, goals, door });
    }
  }
  rooms.sort((a, b) => a.door - b.door);
  return rooms;
}

// ---------- deadlock analysis ----------

// live[i] = a box on cell i can still reach some goal, ignoring other boxes and
// the player's reachability (a relaxation, so pruning on it is sound).
function computeLive(lv: Level, room: Room): boolean[] {
  const live = new Array<boolean>(lv.w * lv.h).fill(false);
  const queue: number[] = [];
  for (const g of room.goals) {
    live[g] = true;
    queue.push(g);
  }
  while (queue.length > 0) {
    const n = queue.pop() as number;
    const nx = n % lv.w;
    const ny = Math.floor(n / lv.w);
    for (const d of DIRS) {
      // c = n - d could have pushed a box into n; the player stood at n - 2d.
      const cx = nx - d.dx;
      const cy = ny - d.dy;
      if (cx < 0 || cy < 0 || cx >= lv.w || cy >= lv.h) continue;
      const c = cy * lv.w + cx;
      if (!room.cellSet.has(c) || live[c]) continue;
      const bx = cx - d.dx;
      const by = cy - d.dy;
      if (bx < 0 || by < 0 || bx >= lv.w || by >= lv.h) continue;
      if (lv.wall[by * lv.w + bx]) continue;
      live[c] = true;
      queue.push(c);
    }
  }
  return live;
}

// A 2x2 block of walls/boxes with no goal and at least one box: every box in it
// is frozen (any push needs the player on the opposite side, which is inside
// the block, or a destination inside the block). Sound to prune. Only the
// blocks containing `cell` are checked: after one push, a new freeze can only
// involve the box that just moved, so checking those blocks is enough.
function hasFreezeAround(lv: Level, boxes: number[], goalSet: Set<number>, cell: number): boolean {
  const boxSet = new Set(boxes);
  const cx = cell % lv.w;
  const cy = Math.floor(cell / lv.w);
  for (let ty = cy - 1; ty <= cy; ty++) {
    for (let tx = cx - 1; tx <= cx; tx++) {
      if (tx < 0 || ty < 0 || tx >= lv.w - 1 || ty >= lv.h - 1) continue;
      const cs = [ty * lv.w + tx, ty * lv.w + tx + 1, (ty + 1) * lv.w + tx, (ty + 1) * lv.w + tx + 1];
      let boxCount = 0;
      let allFilled = true;
      let hasGoal = false;
      for (const c of cs) {
        if (goalSet.has(c)) hasGoal = true;
        if (boxSet.has(c)) boxCount++;
        else if (!lv.wall[c]) allFilled = false;
      }
      if (!hasGoal && allFilled && boxCount > 0) return true;
    }
  }
  return false;
}

// Full-board freeze check, used once on a room's initial boxes.
function hasFreezeAnywhere(lv: Level, boxes: number[], goalSet: Set<number>): boolean {
  for (let y = 0; y < lv.h - 1; y++) {
    for (let x = 0; x < lv.w - 1; x++) {
      if (hasFreezeAround(lv, boxes, goalSet, y * lv.w + x)) return true;
    }
  }
  return false;
}

// ---------- Hungarian assignment (brute force, n <= 4) ----------

function assignmentCost(cost: number[][]): number {
  const n = cost.length;
  if (n === 0) return 0;
  const perm: number[] = [];
  for (let j = 0; j < n; j++) perm.push(j);
  let best = Infinity;
  const rec = (i: number, used: boolean[], sum: number): void => {
    if (sum >= best) return;
    if (i === n) {
      best = sum;
      return;
    }
    for (let j = 0; j < n; j++) {
      if (used[j]) continue;
      used[j] = true;
      rec(i + 1, used, sum + (cost[i] as number[])[j] as number);
      used[j] = false;
    }
  };
  rec(0, new Array<boolean>(n).fill(false), 0);
  return best;
}

// ---------- push-based A* for one room ----------

interface RoomResult {
  moves: string;
  endPlayer: number;
  endBoxes: number[];
}

const ASTAR_CAP = 2_000_000;

function solveRoom(
  lv: Level,
  room: Room,
  startPlayer: number,
  startBoxes: number[],
  fixedBoxes: number[],
): RoomResult | null {
  const goalSet = new Set(room.goals);
  const live = computeLive(lv, room);
  // distFromGoal[gi][cell]: BFS distance from goal gi over the room's cells.
  const distFromGoal: number[][] = room.goals.map((g) => bfsDist(lv, g, room.cellSet));
  const n = startBoxes.length;
  if (n !== room.goals.length) return null;

  const heuristic = (boxes: number[]): number => {
    const cost: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let j = 0; j < n; j++) row.push((distFromGoal[j] as number[])[boxes[i] as number] as number);
      cost.push(row);
    }
    return assignmentCost(cost);
  };

  const isSolved = (boxes: number[]): boolean => boxes.every((b) => goalSet.has(b));
  // `moved` is the destination cell of the box just pushed; a new freeze can
  // only involve it, so only its neighbourhood is checked.
  const isDead = (boxes: number[], moved: number): boolean => {
    for (const b of boxes) {
      if (!goalSet.has(b) && !live[b]) return true;
    }
    return hasFreezeAround(lv, boxes, goalSet, moved);
  };

  if (isSolved(startBoxes)) return { moves: '', endPlayer: startPlayer, endBoxes: startBoxes.slice() };
  if (hasFreezeAnywhere(lv, startBoxes, goalSet)) return null;

  // A* state arrays.
  const sp: number[] = [startPlayer];
  const sb: number[][] = [startBoxes.slice()];
  const sg: number[] = [0];
  const parent: number[] = [-1];
  const seg: string[] = [''];
  const keyOf = (p: number, bs: number[]): string => p + '|' + bs.join(',');
  const best = new Map<string, number>([[keyOf(startPlayer, startBoxes), 0]]);

  // Binary heap of [f, h, counter, state]. Counter makes ties FIFO (deterministic).
  const heap: [number, number, number, number][] = [[heuristic(startBoxes), heuristic(startBoxes), 0, 0]];
  let counter = 1;
  const heapLess = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i] as [number, number, number, number], heap[p] as [number, number, number, number])) break;
      const t = heap[i] as [number, number, number, number];
      heap[i] = heap[p] as [number, number, number, number];
      heap[p] = t;
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0] as [number, number, number, number];
    const last = heap.pop() as [number, number, number, number];
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1] as [number, number, number, number], heap[c] as [number, number, number, number])) c++;
        if (!heapLess(heap[c] as [number, number, number, number], heap[i] as [number, number, number, number])) break;
        const t = heap[i] as [number, number, number, number];
        heap[i] = heap[c] as [number, number, number, number];
        heap[c] = t;
        i = c;
      }
    }
    return top;
  };

  let expanded = 0;
  let solved: number | null = null;
  while (heap.length > 0 && expanded < ASTAR_CAP) {
    const [, , , s] = heapPop();
    if (best.get(keyOf(sp[s] as number, sb[s] as number[])) !== sg[s]) continue; // stale
    const boxes = sb[s] as number[];
    if (isSolved(boxes)) {
      solved = s;
      break;
    }
    expanded++;
    const p = sp[s] as number;
    const blocked = new Set<number>([...boxes, ...fixedBoxes]);
    const { prev, prevDir } = bfsFrom(lv, p, blocked);
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi] as number;
      const bx = b % lv.w;
      const by = Math.floor(b / lv.w);
      for (const d of DIRS) {
        const dest = (by + d.dy) * lv.w + (bx + d.dx);
        if (!room.cellSet.has(dest) || blocked.has(dest)) continue;
        const behind = (by - d.dy) * lv.w + (bx - d.dx);
        if (behind < 0 || behind >= lv.w * lv.h || lv.wall[behind]) continue;
        const walk = pathTo(prev, prevDir, p, behind);
        if (walk === null) continue;
        const nb = boxes.slice();
        nb[bi] = dest;
        nb.sort((a, b2) => a - b2);
        if (isDead(nb, dest)) continue;
        const g2 = (sg[s] as number) + 1;
        const k = keyOf(b, nb);
        const seen = best.get(k);
        if (seen !== undefined && seen <= g2) continue;
        best.set(k, g2);
        const h2 = heuristic(nb);
        const t = sp.length;
        sp.push(b);
        sb.push(nb);
        sg.push(g2);
        parent.push(s);
        seg.push(walk + d.ch);
        heapPush([g2 + h2, h2, counter++, t]);
      }
    }
  }

  if (solved === null) return null;
  let moves = '';
  for (let t = solved; t > 0; t = parent[t] as number) moves = (seg[t] as string) + moves;
  return { moves, endPlayer: sp[solved] as number, endBoxes: sb[solved] as number[] };
}

// Greedy best-first fallback (same successors, priority = heuristic only).
// Finds a valid solution when A* exhausts its cap; not necessarily optimal.
function solveRoomGreedy(
  lv: Level,
  room: Room,
  startPlayer: number,
  startBoxes: number[],
  fixedBoxes: number[],
): RoomResult | null {
  const goalSet = new Set(room.goals);
  const live = computeLive(lv, room);
  const distFromGoal: number[][] = room.goals.map((g) => bfsDist(lv, g, room.cellSet));
  const n = startBoxes.length;
  if (n !== room.goals.length) return null;
  const heuristic = (boxes: number[]): number => {
    const cost: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let j = 0; j < n; j++) row.push((distFromGoal[j] as number[])[boxes[i] as number] as number);
      cost.push(row);
    }
    return assignmentCost(cost);
  };
  const isSolved = (boxes: number[]): boolean => boxes.every((b) => goalSet.has(b));
  const isDead = (boxes: number[], moved: number): boolean => {
    for (const b of boxes) {
      if (!goalSet.has(b) && !live[b]) return true;
    }
    return hasFreezeAround(lv, boxes, goalSet, moved);
  };
  if (isSolved(startBoxes)) return { moves: '', endPlayer: startPlayer, endBoxes: startBoxes.slice() };
  if (hasFreezeAnywhere(lv, startBoxes, goalSet)) return null;

  const sp: number[] = [startPlayer];
  const sb: number[][] = [startBoxes.slice()];
  const parent: number[] = [-1];
  const seg: string[] = [''];
  const keyOf = (p: number, bs: number[]): string => p + '|' + bs.join(',');
  const seen = new Set<string>([keyOf(startPlayer, startBoxes)]);
  const heap: [number, number, number][] = [[heuristic(startBoxes), 0, 0]];
  let counter = 1;
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && a[1] < b[1]);
  const push = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i] as [number, number, number], heap[p] as [number, number, number])) break;
      const t = heap[i] as [number, number, number];
      heap[i] = heap[p] as [number, number, number];
      heap[p] = t;
      i = p;
    }
  };
  const pop = (): [number, number, number] => {
    const top = heap[0] as [number, number, number];
    const last = heap.pop() as [number, number, number];
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1] as [number, number, number], heap[c] as [number, number, number])) c++;
        if (!less(heap[c] as [number, number, number], heap[i] as [number, number, number])) break;
        const t = heap[i] as [number, number, number];
        heap[i] = heap[c] as [number, number, number];
        heap[c] = t;
        i = c;
      }
    }
    return top;
  };

  let expanded = 0;
  while (heap.length > 0 && expanded < ASTAR_CAP) {
    const [, , s] = pop();
    const boxes = sb[s] as number[];
    if (isSolved(boxes)) {
      let moves = '';
      for (let t = s; t > 0; t = parent[t] as number) moves = (seg[t] as string) + moves;
      return { moves, endPlayer: sp[s] as number, endBoxes: boxes.slice() };
    }
    expanded++;
    const p = sp[s] as number;
    const blocked = new Set<number>([...boxes, ...fixedBoxes]);
    const { prev, prevDir } = bfsFrom(lv, p, blocked);
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi] as number;
      const bx = b % lv.w;
      const by = Math.floor(b / lv.w);
      for (const d of DIRS) {
        const dest = (by + d.dy) * lv.w + (bx + d.dx);
        if (!room.cellSet.has(dest) || blocked.has(dest)) continue;
        const behind = (by - d.dy) * lv.w + (bx - d.dx);
        if (behind < 0 || behind >= lv.w * lv.h || lv.wall[behind]) continue;
        const walk = pathTo(prev, prevDir, p, behind);
        if (walk === null) continue;
        const nb = boxes.slice();
        nb[bi] = dest;
        nb.sort((a, b2) => a - b2);
        if (isDead(nb, dest)) continue;
        const k = keyOf(b, nb);
        if (seen.has(k)) continue;
        seen.add(k);
        const t = sp.length;
        sp.push(b);
        sb.push(nb);
        parent.push(s);
        seg.push(walk + d.ch);
        push([heuristic(nb), counter++, t]);
      }
    }
  }
  return null;
}

// ---------- whole level ----------

function solveLevel(text: string): string {
  const lv = parseLevel(text);
  if (lv.boxes.every((b) => lv.goals.includes(b))) return '';
  const rooms = findRooms(lv);
  const assigned = new Set<number>(rooms.flatMap((r) => r.boxes));
  if (lv.boxes.some((b) => !assigned.has(b))) {
    // Unexpected geometry (a box outside every room): treat the whole board as
    // one room and try the same search on it.
    const allCells: number[] = [];
    for (let i = 0; i < lv.w * lv.h; i++) if (!lv.wall[i]) allCells.push(i);
    const whole: Room = {
      cells: allCells,
      cellSet: new Set(allCells),
      boxes: lv.boxes.slice(),
      goals: lv.goals.slice(),
      door: lv.player,
    };
    const r = solveRoom(lv, whole, lv.player, lv.boxes.slice(), []) ?? solveRoomGreedy(lv, whole, lv.player, lv.boxes.slice(), []);
    return r === null ? '' : r.moves;
  }

  let moves = '';
  let player = lv.player;
  let curBoxes = lv.boxes.slice();
  for (const room of rooms) {
    const roomBoxes = curBoxes.filter((b) => room.cellSet.has(b));
    const fixedBoxes = curBoxes.filter((b) => !room.cellSet.has(b));
    const r =
      solveRoom(lv, room, player, roomBoxes, fixedBoxes) ??
      solveRoomGreedy(lv, room, player, roomBoxes, fixedBoxes);
    if (r === null) return ''; // cannot solve this room: no valid answer for the level
    moves += r.moves;
    player = r.endPlayer;
    const rest = curBoxes.filter((b) => !room.cellSet.has(b));
    curBoxes = [...rest, ...r.endBoxes].sort((a, b) => a - b);
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
