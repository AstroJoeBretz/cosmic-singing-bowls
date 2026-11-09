// Chapter 3 widget: tidal stretching of soft vs. stiff stars as the separation changes.
// Illustration: P2 (prolate) deformation with amplitude ~ k / d^3, exaggerated for clarity.
// Both stars in each pair deform toward each other.
(function () {
  var root = document.getElementById("tidew");
  if (!root) return;
  var INK = "#1F4E5A", CLAY = "#B5523B", SOFT = "#4A6F78", SAND = "#C9BBA3";
  var cv = root.querySelector("#tide-canvas"), sl = root.querySelector("#tide-sep");
  var play = root.querySelector("#tide-play"), out = root.querySelector("#tide-readout");
  var D_FAR = 6.0, D_NEAR = 2.6;                 // separation in star radii (centre to centre)
  var ROWS = [{ name: "soft stars", amp: 1.5 }, { name: "stiff stars", amp: 0.45 }];
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function sepOf(v) { return D_FAR + (D_NEAR - D_FAR) * v / 100; }
  function eps(amp, d) { return amp * 3 / (d * d * d); }

  function shade(ctx, cx, cy, rpx, base) {
    var g = ctx.createRadialGradient(cx - 0.3 * rpx, cy - 0.35 * rpx, 0.1 * rpx, cx, cy, 1.5 * rpx);
    g.addColorStop(0, base === INK ? "#4C7E8A" : "#D9806A");
    g.addColorStop(0.55, base);
    g.addColorStop(1, base === INK ? "#12323A" : "#7C3424");
    return g;
  }

  function star(ctx, cx, cy, s, e, base) {
    ctx.beginPath();
    for (var i = 0; i <= 120; i++) {
      var th = i / 120 * 2 * Math.PI, c = Math.cos(th);
      var r = 1 + e * 0.5 * (3 * c * c - 1);               // long axis along x (toward the other star)
      var x = cx + s * r * c, y = cy + s * r * Math.sin(th);
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = shade(ctx, cx, cy, s, base);
    ctx.fill();
  }

  function draw() {
    var d = sepOf(+sl.value);
    var dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    var ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    ctx.font = "600 13px 'Segoe UI', system-ui, sans-serif";
    var rowH = h / ROWS.length, s = Math.min(w / 9.6, rowH / 3.0);
    ROWS.forEach(function (row, i) {
      var cy = rowH * (i + 0.55), e = eps(row.amp, d);
      ctx.fillStyle = INK; ctx.textAlign = "left";
      ctx.fillText(row.name + ": stretched " + Math.round(e * 100) + "%", 8, rowH * i + 20);
      star(ctx, w / 2 - d / 2 * s, cy, s, e, INK);
      star(ctx, w / 2 + d / 2 * s, cy, s, e, CLAY);
      if (i === 0) { ctx.strokeStyle = SAND; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(8, rowH); ctx.lineTo(w - 8, rowH); ctx.stroke(); }
    });
    var label = d > 4.6 ? "far apart" : (d > 3.3 ? "closer" : "about to merge");
    out.textContent = "Separation: " + d.toFixed(1) + " star radii (" + label + "). In this exaggerated illustration, the soft stars stretch about " +
      (ROWS[0].amp / ROWS[1].amp).toFixed(0) + "× as much as the stiff stars.";
  }

  // play / pause: sweep far -> near -> far, easing, with short holds at the ends
  var raf = null, t0 = 0, PERIOD = 9000;
  function setPlaying(on) {
    if (on && !raf) {
      t0 = performance.now() - phaseFromValue(+sl.value) * PERIOD;
      raf = requestAnimationFrame(tick);
    } else if (!on && raf) { cancelAnimationFrame(raf); raf = null; }
    play.textContent = raf ? "Pause animation" : "Play animation";
  }
  function valueFromPhase(p) {                      // p in [0,1): 0 far, .5 near, 1 far
    var u = p < 0.5 ? p * 2 : (1 - p) * 2;
    u = Math.min(1, Math.max(0, (u - 0.08) / 0.84)); // holds at both ends
    return 100 * u * u * (3 - 2 * u);
  }
  function phaseFromValue(v) {                      // inverse on the descending half (approx. fine)
    var lo = 0, hi = 0.5;
    for (var i = 0; i < 24; i++) { var m = (lo + hi) / 2; if (valueFromPhase(m) < v) lo = m; else hi = m; }
    return (lo + hi) / 2;
  }
  function tick(now) {
    var p = ((now - t0) % PERIOD) / PERIOD;
    sl.value = valueFromPhase(p);
    draw();
    raf = requestAnimationFrame(tick);
  }
  play.addEventListener("click", function () { setPlaying(!raf); });
  sl.addEventListener("input", function () { setPlaying(false); draw(); });
  window.addEventListener("resize", draw);

  root.classList.add("ready");
  sl.value = reduce ? 70 : 25;
  draw();
})();
