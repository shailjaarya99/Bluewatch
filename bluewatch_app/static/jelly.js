/* BlueWatch background: one big jellyfish that swims down the page as you scroll,
   a few small ones drifting behind it. Nothing here is data; it is only decoration.
   - scrolling makes the big one pulse harder and its tentacles trail behind
   - moving the mouse nudges it sideways, and it tilts into its turns
   - it pulses fast while the model is working (Jelly.busy) and sends a ripple when a result lands (Jelly.bloom) */
(() => {
  'use strict';
  const cv = document.getElementById('sea');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const TAU = Math.PI * 2;
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const rand = (a, b) => a + Math.random() * (b - a);

  let W = 0, H = 0, dpr = 1;
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = innerWidth; H = innerHeight;
    cv.width = Math.round(W * dpr);
    cv.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  class Jelly {
    constructor(o) {
      Object.assign(this, { x: 0, y: 0, vx: 0, vy: 0, size: 60, alpha: 1, phase: rand(0, TAU),
        rate: 0.9, tilt: 0, par: 0.3, n: 7, segs: 12, c: 0, bw: 0, bh: 0, seed: rand(0, 100) }, o);
      this.setSize(this.size);
      this.tent = [];
      for (let k = 0; k < this.n; k++) {
        const f = this.n === 1 ? 0 : (k / (this.n - 1)) * 2 - 1;
        const pts = [];
        for (let i = 0; i < this.segs; i++) pts.push({ x: this.x + f * this.size * 0.9, y: this.y + i * this.seg });
        this.tent.push({ f, pts, sway: rand(0, TAU), thick: k % 2 === 0 ? 1.5 : 0.9, len: 1 - Math.abs(f) * 0.22 });
      }
    }
    setSize(s) { this.size = s; this.seg = s * 0.3; this.bw = s; this.bh = s * 0.74; }

    pulseShape() {
      const s = Math.sin(this.phase);
      this.c = s > 0 ? s : s * 0.35;           // quick squeeze, slow relax
      this.bw = this.size * (1 - 0.17 * this.c);
      this.bh = this.size * 0.74 * (1 + 0.14 * this.c);
    }

    moveTentacles(dt, t) {
      const ca = Math.cos(this.tilt), sa = Math.sin(this.tilt);
      for (const T of this.tent) {
        const p0 = T.pts[0];
        p0.x = this.x + T.f * this.bw * 0.92 * ca;
        p0.y = this.y + T.f * this.bw * 0.92 * sa + 2;
        const L = this.seg * T.len;
        for (let i = 1; i < T.pts.length; i++) {
          const p = T.pts[i], q = T.pts[i - 1], k = i / T.pts.length;
          p.x += -this.vx * dt * 0.55 * k + Math.sin(t * 1.7 + T.sway + i * 0.6) * dt * this.size * 0.9 * k;
          p.y += -this.vy * dt * 0.55 * k + dt * this.size * 0.5;
          let dx = (p.x - q.x) * 0.94, dy = p.y - q.y;
          if (dy < L * 0.15) dy = L * 0.15;
          const d = Math.hypot(dx, dy) || 1;
          p.x = q.x + (dx / d) * L;
          p.y = q.y + (dy / d) * L;
        }
      }
    }

    draw() {
      const A = this.alpha, { bw, bh } = this;
      ctx.lineCap = 'round';
      for (const T of this.tent) {
        const n = T.pts.length;
        for (let i = 1; i < n; i++) {
          const a = T.pts[i - 1], b = T.pts[i], k = i / n;
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
          ctx.lineWidth = Math.max(0.5, T.thick * this.size * 0.045 * (1 - k * 0.75));
          ctx.strokeStyle = `rgba(223,182,178,${0.5 * (1 - k * 0.9) * A})`;
          ctx.stroke();
        }
      }
      ctx.save();
      ctx.translate(this.x, this.y);
      ctx.rotate(this.tilt);
      // soft glow around the bell
      const halo = ctx.createRadialGradient(0, -bh * 0.4, 0, 0, -bh * 0.4, bw * 2.3);
      halo.addColorStop(0, `rgba(223,182,178,${0.16 * A})`);
      halo.addColorStop(1, 'rgba(223,182,178,0)');
      ctx.fillStyle = halo;
      ctx.fillRect(-bw * 2.4, -bh * 2.4, bw * 4.8, bh * 4.2);
      // bell: dome on top, scalloped edge underneath
      ctx.beginPath();
      ctx.moveTo(-bw, 0);
      ctx.bezierCurveTo(-bw, -bh * 1.35, bw, -bh * 1.35, bw, 0);
      const sc = 7;
      for (let i = 0; i < sc; i++) {
        const x0 = bw - (2 * bw * i) / sc, x1 = bw - (2 * bw * (i + 1)) / sc;
        ctx.quadraticCurveTo((x0 + x1) / 2, bh * 0.2, x1, 0);
      }
      ctx.closePath();
      const body = ctx.createRadialGradient(0, -bh * 0.55, 0, 0, -bh * 0.45, bw * 1.25);
      body.addColorStop(0, `rgba(251,228,216,${0.36 * A})`);
      body.addColorStop(0.55, `rgba(223,182,178,${0.2 * A})`);
      body.addColorStop(1, `rgba(133,79,108,${0.34 * A})`);
      ctx.fillStyle = body;
      ctx.fill();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = `rgba(251,228,216,${0.5 * A})`;
      ctx.stroke();
      // ribs and the glowing ring inside the bell
      ctx.lineWidth = 1;
      ctx.strokeStyle = `rgba(251,228,216,${0.14 * A})`;
      for (let k = -3; k <= 3; k++) {
        ctx.beginPath();
        ctx.moveTo(0, -bh * 1.02);
        ctx.quadraticCurveTo(k * bw * 0.34, -bh * 0.55, k * bw * 0.3, -bh * 0.04);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.ellipse(0, -bh * 0.42, bw * 0.4, bh * 0.2, 0, 0, TAU);
      ctx.strokeStyle = `rgba(251,228,216,${0.38 * A})`;
      ctx.stroke();
      ctx.restore();
    }
  }

  // ---- scene ----
  let main, small = [], rings = [];
  const mainSize = () => clamp(W * 0.075, 42, 78);
  function build() {
    main = new Jelly({ x: W * 0.66, y: H * 0.36, size: mainSize(), n: 9, segs: 17, alpha: 1 });
    small = [];
    const count = W < 700 ? 3 : 5;
    for (let i = 0; i < count; i++) {
      const s = rand(15, 32);
      small.push(new Jelly({ x: rand(0.05, 0.95) * W, y: rand(0.1, 1.1) * H, size: s, n: 5, segs: 10,
        alpha: rand(0.3, 0.55), par: rand(0.12, 0.5), rate: rand(0.55, 0.9) }));
    }
  }

  // ---- inputs ----
  let lastY = scrollY, dScroll = 0, sv = 0, prog = 0, mx = 0.5, my = 0.5, busy = false, boost = 0;
  function onScroll() {
    const y = scrollY;
    dScroll += y - lastY; lastY = y;
    const max = document.documentElement.scrollHeight - innerHeight;
    prog = max > 0 ? clamp(y / max, 0, 1) : 0;
    document.documentElement.style.setProperty('--p', prog.toFixed(4));
  }
  addEventListener('scroll', onScroll, { passive: true });
  addEventListener('pointermove', e => { mx = e.clientX / W; my = e.clientY / H; }, { passive: true });
  addEventListener('resize', () => { resize(); if (main) main.setSize(mainSize()); if (reduce) still(); });

  window.Jelly = {
    busy(b) { busy = !!b; },
    bloom() {
      if (reduce || !main) return;
      boost = 1;
      rings.push({ x: main.x, y: main.y - main.bh * 0.4, r: main.size * 0.8, a: 0.55 });
    },
  };

  // ---- loop ----
  let last = performance.now(), time = 0;
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now; time += dt;
    const ds = dScroll; dScroll = 0;
    sv = lerp(sv, dt > 0 ? ds / dt : 0, 0.12);
    const speed = clamp(Math.abs(sv) / 1400, 0, 1.5);
    boost = Math.max(0, boost - dt * 0.7);

    // big jellyfish: follows the scroll position down the page, sways side to side
    const tx = W * (0.5 + 0.3 * Math.sin(prog * TAU * 1.5 + 0.55)) + (mx - 0.5) * W * 0.1;
    const ty = H * (0.34 + 0.34 * prog) + (my - 0.5) * H * 0.06;
    main.vx += ((tx - main.x) * 3.0 - main.vx * 2.3) * dt;
    main.vy += ((ty - main.y) * 2.4 - main.vy * 2.3) * dt + clamp(sv, -1800, 1800) * dt * 0.9;
    main.x += main.vx * dt;
    main.y += main.vy * dt;
    main.tilt = lerp(main.tilt, clamp(main.vx * 0.0016, -0.45, 0.45), 0.08);
    main.rate = 0.85 + speed * 1.5 + boost * 1.4 + (busy ? 1.7 : 0);
    main.phase += dt * TAU * main.rate;
    main.pulseShape();
    main.moveTentacles(dt, time);

    // small ones: drift upward, a little slower than the page so you feel depth
    for (const j of small) {
      j.phase += dt * TAU * (j.rate + speed * 0.6);
      j.pulseShape();
      j.vx = Math.sin(time * 0.3 + j.seed) * 10;
      j.vy = -(8 + 26 * Math.max(0, j.c)) * (j.size / 24);
      j.x += j.vx * dt;
      j.y += j.vy * dt - ds * j.par;
      j.tilt = lerp(j.tilt, clamp(j.vx * 0.01, -0.3, 0.3), 0.05);
      if (j.y < -j.size * 6) { j.y = H + j.size * 3; j.x = rand(0.05, 0.95) * W; }
      else if (j.y > H + j.size * 9) { j.y = -j.size * 4; j.x = rand(0.05, 0.95) * W; }
      j.moveTentacles(dt, time);
    }

    paint();
    requestAnimationFrame(frame);
  }

  function paint() {
    ctx.clearRect(0, 0, W, H);
    ctx.globalCompositeOperation = 'lighter';
    for (const j of small) j.draw();
    main.draw();
    for (let i = rings.length - 1; i >= 0; i--) {
      const r = rings[i];
      r.r += 7; r.a -= 0.011;
      if (r.a <= 0) { rings.splice(i, 1); continue; }
      ctx.beginPath(); ctx.arc(r.x, r.y, r.r, 0, TAU);
      ctx.lineWidth = 2; ctx.strokeStyle = `rgba(223,182,178,${r.a})`; ctx.stroke();
    }
    ctx.globalCompositeOperation = 'source-over';
  }

  // one still frame for people who asked their system for less motion
  function still() {
    main.pulseShape();
    for (let i = 0; i < 30; i++) main.moveTentacles(0.016, 0);
    for (const j of small) { j.pulseShape(); for (let i = 0; i < 30; i++) j.moveTentacles(0.016, 0); }
    paint();
  }

  resize();
  build();
  onScroll();
  if (reduce) still(); else requestAnimationFrame(t => { last = t; frame(t); });
})();
