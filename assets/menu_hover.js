// Abre los desplegables del menú superior al pasar el cursor (solo en
// dispositivos con puntero fino). El clic sigue funcionando como siempre.
// El cierre espera un momento para que el cursor pueda llegar al menú o
// volver a él sin que se cierre.
(function () {
  if (!window.matchMedia || !window.matchMedia("(hover: hover) and (pointer: fine)").matches) {
    return;
  }

  const RETARDO_CIERRE = 300; // ms
  const temporizadores = new WeakMap();

  function alternar(bloque, abrir) {
    const boton = bloque.querySelector(".dropdown-toggle");
    if (!boton) return;
    const abierto = boton.getAttribute("aria-expanded") === "true";
    if (abierto !== abrir) boton.click();
  }

  function cancelarCierre(bloque) {
    clearTimeout(temporizadores.get(bloque));
    temporizadores.delete(bloque);
  }

  document.addEventListener("mouseover", function (evento) {
    const bloque = evento.target.closest(".nav-bloque");
    if (!bloque) return;
    cancelarCierre(bloque);
    if (!bloque.contains(evento.relatedTarget)) alternar(bloque, true);
  });

  document.addEventListener("mouseout", function (evento) {
    const bloque = evento.target.closest(".nav-bloque");
    if (!bloque || bloque.contains(evento.relatedTarget)) return;
    cancelarCierre(bloque);
    temporizadores.set(bloque, setTimeout(function () {
      temporizadores.delete(bloque);
      alternar(bloque, false);
    }, RETARDO_CIERRE));
  });
})();
