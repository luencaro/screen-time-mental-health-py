// Rejilla tipo mosaico para .rejilla-mosaico: cada hijo ocupa tantas filas
// de 4 px como mide su contenido, así las cards no se estiran a la altura de
// su vecina. Se recalcula cuando cambia el tamaño de una card (p. ej. al
// terminar MathJax o al cambiar el ancho de la ventana).
(function () {
  const FILA = 4; // px, igual que grid-auto-rows en componentes.css

  function ajustar(item) {
    const margen = parseFloat(getComputedStyle(item).marginBottom) || 0;
    const alto = item.getBoundingClientRect().height + margen;
    item.style.gridRowEnd = "span " + Math.ceil(alto / FILA);
  }

  const observador = new ResizeObserver(function (entradas) {
    entradas.forEach(function (e) { ajustar(e.target); });
  });

  function preparar(rejilla) {
    if (rejilla.dataset.mosaico) return;
    rejilla.dataset.mosaico = "1";
    Array.from(rejilla.children).forEach(function (item) {
      ajustar(item);
      observador.observe(item);
    });
    rejilla.classList.add("mosaico-activo");
  }

  new MutationObserver(function () {
    document.querySelectorAll(".rejilla-mosaico").forEach(preparar);
  }).observe(document.body, { childList: true, subtree: true });
})();
