// Widget #2: how loud must the signal be for the bias to stand out?
// Rescales the real SNR-500 posterior: width ~ 1/SNR, bias held fixed (approximation).
(function () {
  var root = document.getElementById("snrw");
  if (!root) return;
  var CLAY = "#B5523B", INK = "#1F4E5A", SOFT = "#4A6F78", SAND = "#C9BBA3", GOLD = "#C18A2C";
  var cv = root.querySelector("#snr-plot"), sl = root.querySelector("#snr-slider"), out = root.querySelector("#snr-readout");
  var D = null;
  fetch("assets/snr_posterior.json").then(function (r) { return r.json(); }).then(function (d) {
    D = d; root.classList.add("ready"); draw();
  }).catch(function () { root.classList.add("failed"); });

  function ordinal(n) {
    var r = n % 100, s = (r >= 11 && r <= 13) ? "th" : ({ 1: "st", 2: "nd", 3: "rd" }[n % 10] || "th");
    return n + s;
  }
  function snr() { return 50 * Math.pow(20, (+sl.value) / 100); }   // 50 .. 1000, log scale

  function percentileOfTruth(k) {
    // fraction of rescaled samples below the true value
    var x = (D.truth - D.best_fit) / k, q = D.dev_quantiles, lo = 0, hi = q.length - 1;
    if (x <= q[0]) return 0; if (x >= q[hi]) return 100;
    while (hi - lo > 1) { var m = (lo + hi) >> 1; if (q[m] < x) lo = m; else hi = m; }
    return 100 * (lo + (x - q[lo]) / (q[hi] - q[lo])) / (q.length - 1);
  }

  function draw() {
    if (!D) return;
    var s = snr(), k = D.snr_ref / s;
    var dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    var ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    ctx.font = "12px 'Segoe UI', system-ui, sans-serif";
    var L = 12, R = w - 12, T = 14, B = h - 34, xlo = 280, xhi = 580;
    function X(v) { return L + (v - xlo) / (xhi - xlo) * (R - L); }
    var pk = 0, pdf = D.dev_pdf.map(function (p) { return p / k; });
    pdf.forEach(function (p) { if (p > pk) pk = p; });
    // fixed y scale so narrowing is visible: tallest curve is at SNR 1000
    var yfix = (Math.max.apply(null, D.dev_pdf) / (D.snr_ref / 1000)) * 1.1;
    function Y(p) { return B - p / yfix * (B - T); }
    ctx.strokeStyle = INK; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(L, B); ctx.lineTo(R, B); ctx.stroke();
    ctx.fillStyle = SOFT; ctx.textAlign = "center";
    for (var t = 300; t <= 560; t += 50) { ctx.fillText(t, X(t), B + 15); }
    ctx.fillText("measured tidal deformability", (L + R) / 2, h - 4);
    ctx.beginPath(); ctx.moveTo(X(D.best_fit + D.dev_grid[0] * k), B);
    for (var i = 0; i < D.dev_grid.length; i++) ctx.lineTo(X(D.best_fit + D.dev_grid[i] * k), Y(pdf[i]));
    ctx.lineTo(X(D.best_fit + D.dev_grid[D.dev_grid.length - 1] * k), B); ctx.closePath();
    ctx.fillStyle = "rgba(31,78,90,.22)"; ctx.fill();
    ctx.strokeStyle = INK; ctx.lineWidth = 2.5; ctx.beginPath();
    for (i = 0; i < D.dev_grid.length; i++) { var px = X(D.best_fit + D.dev_grid[i] * k), py = Y(pdf[i]); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); }
    ctx.stroke();
    ctx.strokeStyle = GOLD; ctx.lineWidth = 2; ctx.setLineDash([6, 4]); ctx.beginPath(); ctx.moveTo(X(D.truth), T); ctx.lineTo(X(D.truth), B); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#8A5F10"; ctx.textAlign = "right"; ctx.fillText("true value", X(D.truth) - 6, T + 10);
    ctx.fillStyle = CLAY; ctx.textAlign = "left"; ctx.fillText("what the textbook model reports", X(D.best_fit) + 8, T + 10);
    ctx.fillStyle = CLAY; ctx.beginPath(); ctx.arc(X(D.best_fit), B, 5, 0, 6.2832); ctx.fill();

    var sigma = D.sigma * k, bias = D.best_fit - D.truth, ratio = bias / sigma, pc = percentileOfTruth(k);
    root.querySelector("#snr-val").textContent = Math.round(s);
    var verdict = ratio < 1 ? "The error hides inside the noise: you would not know anything was wrong."
      : (ratio < 2 ? "The error is starting to show." : "The error is clear: the true value is well outside what the data allow.");
    out.innerHTML = "Uncertainty about ±" + sigma.toFixed(0) + ", but the answer is off by +" + bias.toFixed(0) +
      " (" + ratio.toFixed(1) + " times the uncertainty). The true value sits " + (pc < 1 ? "below the 1st percentile of what the data allow" : "at the " + ordinal(Math.round(pc)) + " percentile") +
      ". <strong>" + verdict + "</strong>";
  }
  sl.addEventListener("input", draw);
  window.addEventListener("resize", draw);
})();
