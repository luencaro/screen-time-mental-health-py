(function () {
  const posiciones = new Map();
  const movimientoReducido = window.matchMedia("(prefers-reduced-motion: reduce)");
  let rutaActiva;
  let restaurar;

  // El historial cambia antes de que Dash tenga listo el contenido de destino.
  if ("scrollRestoration" in history) history.scrollRestoration = "manual";

  function recordarPosicion() {
    if (rutaActiva === window.location.pathname) posiciones.set(rutaActiva, window.scrollY);
  }

  window.addEventListener("scroll", recordarPosicion, {passive: true});
  document.addEventListener("click", (evento) => {
    if (!evento.target.closest("a[href]")) return;
    recordarPosicion();
    restaurar = undefined;
    const origen = rutaActiva;
    const posicion = window.scrollY;
    // Los enlaces de Dash llevan al inicio de inmediato; conserva la vista anterior hasta el cambio.
    if (origen && posicion) requestAnimationFrame(() => {
      if (rutaActiva === origen && window.location.pathname !== origen) window.scrollTo(0, posicion);
    });
  }, true);
  window.addEventListener("popstate", (evento) => {
    if (!evento.isTrusted || rutaActiva === window.location.pathname) return;
    if (rutaActiva) posiciones.set(rutaActiva, window.scrollY);
    restaurar = window.location.pathname;
  });
  function presentar(vista) {
    if (!vista || vista.ruta !== window.location.pathname) return window.dash_clientside.no_update;
    document.title = vista.titulo;
    if (rutaActiva === vista.ruta) return vista.ruta;

    const veniaDeOtraVista = Boolean(rutaActiva);
    const volver = restaurar === vista.ruta;
    const posicion = volver ? posiciones.get(vista.ruta) || 0 : 0;
    rutaActiva = vista.ruta;
    restaurar = undefined;

    requestAnimationFrame(() => {
      if (vista.ruta !== window.location.pathname) return;
      if (veniaDeOtraVista && !window.location.hash) window.scrollTo(0, posicion);
      recordarPosicion();
      if (veniaDeOtraVista && !document.querySelector("dialog[open]")) {
        document.getElementById("contenido-principal")?.focus({preventScroll: true});
      }
      if (!movimientoReducido.matches && !volver) {
        document.getElementById("contenido")?.animate?.(
          [{opacity: 0.75, translate: "0 5px"}, {opacity: 1, translate: "0 0"}],
          {duration: 230, easing: "cubic-bezier(0.2, 0.65, 0.25, 1)"},
        );
      }
    });
    return vista.ruta;
  }

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    navegacion: {presentar},
  });
})();
