(() => {
  const benches = {
    imagenet: {family:'Vision classifier',title:'ImageNet-C · ViT-B/16',context:'Corrupted natural images · severity 5',source:55.5,beta:62.6},
    eurosat: {family:'Vision-language model',title:'EuroSAT · CLIP ViT-B/16',context:'Satellite land-cover classification',source:42.0,beta:53.3},
    dermclip: {family:'Vision-language model',title:'Derm7pt · CLIP ViT-B/16',context:'Medical skin-lesion classification · 7 classes',source:55.9,beta:58.6},
    dermbiomed: {family:'Vision-language model',title:'Derm7pt · BiomedCLIP',context:'Medical target · natural-image CLIP steering',source:60.9,beta:62.1}
  };
  const tabs = [...document.querySelectorAll('[data-benchmark]')];
  function select(button) {
    const d = benches[button.dataset.benchmark];
    tabs.forEach(t => {t.setAttribute('aria-selected', String(t === button));t.tabIndex = t === button ? 0 : -1;});
    for (const [id, value] of Object.entries({'benchmark-family':d.family,'benchmark-title':d.title,'benchmark-context':d.context,'source-value':d.source.toFixed(1),'beta-value':d.beta.toFixed(1),'benchmark-gain':`+${(d.beta-d.source).toFixed(1)} percentage points`})) document.getElementById(id).textContent = value;
    document.getElementById('source-bar').style.width = `${d.source}%`;
    document.getElementById('beta-bar').style.width = `${d.beta}%`;
  }
  tabs.forEach((t,i) => {
    t.tabIndex = i === 0 ? 0 : -1;
    t.addEventListener('click', () => select(t));
    t.addEventListener('keydown', e => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      e.preventDefault();const n = tabs[(i+(e.key === 'ArrowRight'?1:tabs.length-1))%tabs.length];select(n);n.focus();
    });
  });
  document.querySelectorAll('.research-loop').forEach(video => {
    video.muted = true;video.defaultMuted = true;video.loop = true;video.playsInline = true;video.controls = false;
    function fallback() {
      if (video.dataset.fallbackShown) return;
      video.dataset.fallbackShown = 'true';const image = document.createElement('img');
      image.src = video.dataset.gif;image.className = 'loop-fallback';image.alt = video.getAttribute('aria-label');
      video.hidden = true;video.style.display = 'none';video.parentElement.appendChild(image);
    }
    video.addEventListener('error',fallback);
    video.play().catch(fallback);
  });
})();
