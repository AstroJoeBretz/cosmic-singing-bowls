// Scroll progress, reveal-on-scroll, and animation controls. Everything degrades gracefully without JS.
document.documentElement.classList.add("js");

(function () {
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // progress bar
  var bar = document.getElementById("progress");
  function onScroll() {
    var h = document.documentElement;
    var max = h.scrollHeight - h.clientHeight;
    if (bar && max > 0) bar.style.width = (100 * h.scrollTop / max) + "%";
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // reveal on scroll
  var items = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) {
    items.forEach(function (el) { el.classList.add("in"); });
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
      });
    }, { threshold: 0.12 });
    items.forEach(function (el) { io.observe(el); });
  }

  // animations: pause/play button, pause off-screen, no autoplay if motion is reduced
  document.querySelectorAll("video").forEach(function (v) {
    var userPaused = reduce;
    if (reduce) { v.removeAttribute("autoplay"); v.pause(); }
    var bar = document.createElement("div");
    bar.className = "vid-controls";
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "vid-toggle";
    function label() { btn.textContent = v.paused ? "Play animation" : "Pause animation"; }
    btn.addEventListener("click", function () {
      if (v.paused) { userPaused = false; v.play(); } else { userPaused = true; v.pause(); }
      label();
    });
    v.addEventListener("play", label);
    v.addEventListener("pause", label);
    label();
    bar.appendChild(btn);

    // Videos with a soundtrack: muted by default, with a sound button and volume slider.
    if (v.hasAttribute("data-audio")) {
      v.muted = true;
      v.volume = 0.6;
      var snd = document.createElement("button");
      snd.type = "button";
      snd.className = "vid-toggle";
      var vol = document.createElement("input");
      vol.type = "range"; vol.min = "0"; vol.max = "100"; vol.step = "1"; vol.value = "60";
      vol.className = "vid-volume";
      vol.setAttribute("aria-label", "Volume");
      var sndLabel = function () {
        snd.textContent = v.muted ? "Sound off" : "Sound on";
      };
      snd.addEventListener("click", function () {
        if (v.muted && +vol.value === 0) { vol.value = "60"; v.volume = 0.6; }
        v.muted = !v.muted; sndLabel();
      });
      vol.addEventListener("input", function () {
        v.volume = (+vol.value) / 100;
        v.muted = +vol.value === 0;          // dragging up from zero or while muted turns sound on
        if (+vol.value > 0) v.muted = false;
        sndLabel();
      });
      sndLabel();
      bar.appendChild(snd);
      bar.appendChild(vol);
    }
    var cap = v.parentNode.querySelector("figcaption");
    v.parentNode.insertBefore(bar, cap);
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (userPaused) return;
          if (e.isIntersecting) v.play().catch(function () {}); else v.pause();
        });
      }, { threshold: 0.25 }).observe(v);
    }
  });
})();
