// Publica el alto real del menú superior (fijo) en la variable CSS --alto-cabecera, para que los
// elementos fijos de las secciones (p. ej. el filtro de modelos de Benchmarking) queden justo debajo.
// En pantallas estrechas el menú pasa a dos filas, por eso se mide en vez de usar un valor fijo.
(function () {
  function medir() {
    const cabecera = document.querySelector(".cabecera");
    if (cabecera) {
      document.documentElement.style.setProperty("--alto-cabecera", cabecera.offsetHeight + "px");
    }
  }
  function iniciar() {
    const cabecera = document.querySelector(".cabecera");
    if (!cabecera) { setTimeout(iniciar, 100); return; }   // Dash aún no ha dibujado el layout
    medir();
    if (window.ResizeObserver) new ResizeObserver(medir).observe(cabecera);
    window.addEventListener("resize", medir);
  }
  iniciar();
})();
