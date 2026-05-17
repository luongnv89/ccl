// ── Fetch GitHub stars ─────────────────────────────────────
fetch("https://api.github.com/repos/luongnv89/claude-codex-local")
  .then((res) => res.json())
  .then((data) => {
    const starCount = data.stargazers_count || 0;
    const starEl = document.getElementById("star-count");
    if (starEl) {
      starEl.textContent = `⭐ ${starCount} stars`;
    }
  })
  .catch(() => {
    const starEl = document.getElementById("star-count");
    if (starEl) {
      starEl.textContent = "⭐ Star on GitHub";
    }
  });

// ── Scroll reveal ──────────────────────────────────────────
const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        revealObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.1, rootMargin: "0px 0px -40px 0px" },
);

document
  .querySelectorAll(".reveal")
  .forEach((el) => revealObserver.observe(el));

// ── Active nav link ────────────────────────────────────────
const sections = [
  "hero",
  "problem",
  "features",
  "how-it-works",
  "install",
  "faq",
  "changelog",
  "cta",
];
const navLinks = document.querySelectorAll(".nav-links a");

// Only set up section observer if any of the sections exist (index.html only)
if (sections.some((id) => document.getElementById(id))) {
  const sectionObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const id = entry.target.id;
          navLinks.forEach((a) => {
            a.classList.toggle(
              "active",
              a.getAttribute("href") === "#" + id,
            );
          });
        }
      });
    },
    { threshold: 0.4 },
  );

  sections.forEach((id) => {
    const el = document.getElementById(id);
    if (el) sectionObserver.observe(el);
  });
}

// ── Copy buttons ───────────────────────────────────────────
document.querySelectorAll(".copy-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const text = btn.dataset.copy;
    navigator.clipboard
      .writeText(text)
      .then(() => {
        const original = btn.innerHTML;
        btn.innerHTML = `<svg viewBox="0 0 24 24" style="width:13px;height:13px;fill:none;stroke:currentColor;stroke-width:1.8"><polyline points="20 6 9 17 4 12"/></svg> Copied!`;
        btn.classList.add("copied");
        setTimeout(() => {
          btn.innerHTML = original;
          btn.classList.remove("copied");
        }, 2000);
      })
      .catch(() => {
        // Fallback for older browsers
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
      });
  });
});

// ── FAQ accordion (legacy + terminal-brutalism variants) ───
document
  .querySelectorAll(".faq-question, .tb-faq-q")
  .forEach((btn) => {
    btn.addEventListener("click", () => {
      const item = btn.closest(".faq-item, .tb-faq-item");
      if (!item) return;
      const isOpen = item.classList.contains("open");

      const groupSelector = item.classList.contains("tb-faq-item")
        ? ".tb-faq-item.open"
        : ".faq-item.open";

      document
        .querySelectorAll(groupSelector)
        .forEach((el) => el.classList.remove("open"));

      if (!isOpen) item.classList.add("open");
    });
  });

// ── Sidebar scroll-spy for docs/guides pages ───────────────
const sidebarLinks = document.querySelectorAll(".tb-sidebar-list a[href^='#']");
if (sidebarLinks.length) {
  const targetMap = new Map();
  sidebarLinks.forEach((a) => {
    const id = a.getAttribute("href").slice(1);
    const el = document.getElementById(id);
    if (el) targetMap.set(el, a);
  });

  const setActive = (link) => {
    sidebarLinks.forEach((a) => a.classList.remove("active"));
    if (link) link.classList.add("active");
  };

  const spyObserver = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((e) => e.isIntersecting)
        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (visible.length) {
        const link = targetMap.get(visible[0].target);
        setActive(link);
      }
    },
    { rootMargin: "-20% 0px -65% 0px", threshold: 0 },
  );

  targetMap.forEach((_, el) => spyObserver.observe(el));
}
