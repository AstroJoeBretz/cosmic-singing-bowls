// Widget #1: "Retune the bowl". Interpolates a grid precomputed with the paper's waveform code.
(function () {
  var root = document.getElementById("retune");
  if (!root) return;
  var CLAY = "#B5523B", INK = "#1F4E5A", SOFT = "#4A6F78", SAND = "#C9BBA3", PANEL = "#F2ECE2", BG = "#FAF7F2";
  var dial = root.querySelector("#rt-dial"), plot = root.querySelector("#rt-plot");
  var sSpin = root.querySelector("#rt-spin"), sNl = root.querySelector("#rt-nl"), cGr = root.querySelector("#rt-gr");
  var out = root.querySelector("#rt-readout");
  var G = null;

  fetch("assets/trc_grid.json").then(function (r) { return r.json(); }).then(function (g) {
    G = g; root.classList.add("ready"); draw();
  }).catch(function () { root.classList.add("failed"); });

  // locate fractional index of v on ascending/descending array
  function locate(arr, v) {
    var n = arr.length, asc = arr[n - 1] > arr[0];
    for (var i = 0; i < n - 1; i++) {
      var a = arr[i], b = arr[i + 1];
      if ((asc && v <= b) || (!asc && v >= b)) return [i, (v - a) / (b - a)];
    }
    return [n - 2, 1];
  }
  function idx(gr, i, j) { return (gr * G.S.length + i) * G.Anl.length + j; }
  function interp(table, gr, S, A) {
    var li = locate(G.S, S), lj = locate(G.Anl, A);
    var i = li[0], j = lj[0], u = li[1], v = lj[1];
    function at(ii, jj) { return table[idx(gr, ii, jj)]; }
    if (Array.isArray(at(0, 0))) {
      var n = at(0, 0).length, res = new Array(n);
      for (var k = 0; k < n; k++) {
        res[k] = (1 - u) * ((1 - v) * at(i, j)[k] + v * at(i, j + 1)[k]) +
                 u * ((1 - v) * at(i + 1, j)[k] + v * at(i + 1, j + 1)[k]);
      }
      return res;
    }
    return (1 - u) * ((1 - v) * at(i, j) + v * at(i, j + 1)) + u * ((1 - v) * at(i + 1, j) + v * at(i + 1, j + 1));
  }

  function setup(c) {
    var dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    }
    var ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.font = "12px 'Segoe UI', system-ui, sans-serif";
    return { ctx: ctx, w: w, h: h };
  }

  function drawDial(res, base) {
    var s = setup(dial), ctx = s.ctx, w = s.w, h = s.h;
    var lo = 1300, hi = 2300, L = 16, R = w - 16, y = h * 0.52;
    function X(f) { return L + (f - lo) / (hi - lo) * (R - L); }
    ctx.fillStyle = PANEL; ctx.fillRect(L, y - 3, R - L, 6);
    // inspiral band that the signal actually reaches
    ctx.fillStyle = "rgba(31,78,90,.18)"; ctx.fillRect(L, y - 3, X(G.f_merger) - L, 6);
    ctx.textAlign = "center";
    ctx.fillStyle = INK;
    ctx.fillRect(X(G.f_merger) - 1, y - 14, 2, 28);
    ctx.fillText("merger (1553 Hz)", X(G.f_merger), y - 20);
    ctx.fillStyle = SOFT;
    ctx.fillRect(X(base) - 1, y + 2, 2, 16);
    ctx.fillText("textbook pitch", X(base), y + 33);
    var x = X(Math.max(lo, Math.min(hi, res)));
    ctx.fillStyle = CLAY;
    ctx.beginPath(); ctx.arc(x, y, 9, 0, 6.2832); ctx.fill();
    ctx.fillText("this star's pitch", x, y - 20 - (Math.abs(x - X(G.f_merger)) < 90 ? 16 : 0));
    ctx.textAlign = "left"; ctx.fillStyle = SOFT;
    ctx.fillText("← before merger", L, h - 4);
    ctx.textAlign = "right"; ctx.fillText("beyond merger →", R, h - 4);
  }

  function drawPlot(curve) {
    var s = setup(plot), ctx = s.ctx, w = s.w, h = s.h;
    var L = 46, R = w - 12, T = 12, B = h - 34;
    var mn = Math.min.apply(null, curve), mx = Math.max.apply(null, curve);
    var lo = Math.min(-0.5, mn * 1.12), hi = Math.max(0.5, mx * 1.12);
    var f0 = G.f[0], f1 = G.f[G.f.length - 1];
    function X(f) { return L + (f - f0) / (f1 - f0) * (R - L); }
    function Y(v) { return T + (hi - v) / (hi - lo) * (B - T); }
    ctx.strokeStyle = SAND; ctx.lineWidth = 1; ctx.fillStyle = SOFT; ctx.textAlign = "right";
    var step = (hi - lo) > 3 ? 1 : 0.5;
    for (var v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
      ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(R, Y(v)); ctx.globalAlpha = Math.abs(v) < 1e-9 ? 1 : 0.35; ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillText(v.toFixed(1), L - 6, Y(v) + 4);
    }
    ctx.textAlign = "center";
    [900, 1100, 1300, 1500].forEach(function (f) { ctx.fillText(f, X(f), B + 16); });
    ctx.fillText("gravitational-wave frequency [Hz]", (L + R) / 2, h - 4);
    ctx.save(); ctx.translate(12, (T + B) / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText("phase change vs. textbook model [rad]", 0, 0); ctx.restore();
    ctx.strokeStyle = CLAY; ctx.lineWidth = 3; ctx.lineJoin = "round"; ctx.beginPath();
    for (var i = 0; i < curve.length; i++) {
      var px = X(G.f[i]), py = Y(curve[i]);
      if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py);
    }
    ctx.stroke();
    // Label the zero line at the right end, where the retuned curve has drifted furthest away,
    // on the side of the line opposite the curve so the two never overlap.
    var endv = curve[curve.length - 1];
    ctx.textAlign = "right"; ctx.fillStyle = INK;
    ctx.fillText("textbook model", R - 6, endv >= 0 ? Y(0) + 16 : Y(0) - 6);
  }

  function state() {
    return { S: +sSpin.value, A: -0.03 * (+sNl.value) / 100, gr: cGr.checked ? 1 : 0 };
  }

  function draw() {
    if (!G) return;
    var st = state();
    var curve = interp(G.curves, st.gr, st.S, st.A);
    var res = interp(G.f_res, st.gr, st.S, st.A);
    drawDial(res, G.f_res_base);
    drawPlot(curve);
    var end = curve[curve.length - 1];
    var spinTxt = st.S === 0 ? "not spinning" : (Math.abs(st.S) + " Hz " + (st.S < 0 ? "against" : "with") + " the orbit");
    root.querySelector("#rt-spin-val").textContent = spinTxt;
    root.querySelector("#rt-nl-val").textContent = (+sNl.value === 0 ? "off" : (+sNl.value) + "% of typical");
    var shift = res - G.f_res_base;
    out.textContent = "Pitch " + (shift === 0 ? "unchanged" : (shift < 0 ? "lowered" : "raised") + " by " + Math.abs(Math.round(shift)) + " Hz") +
      "; by merger the signal's phase is off by " + Math.abs(end).toFixed(2) + " rad (" + (end < 0 ? "behind" : "ahead of") + " the textbook model).";
  }

  [sSpin, sNl, cGr].forEach(function (el) { el.addEventListener("input", draw); });
  root.querySelectorAll("[data-preset]").forEach(function (b) {
    b.addEventListener("click", function () {
      var p = b.getAttribute("data-preset").split(",");
      sSpin.value = p[0]; sNl.value = p[1]; cGr.checked = p[2] === "1"; draw();
    });
  });
  window.addEventListener("resize", draw);
})();
