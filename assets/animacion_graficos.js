// Animación de entrada de los gráficos: la primera vez que un gráfico se
// dibuja con datos, sus barras y líneas crecen desde cero. Los ejes se fijan
// durante la animación para que no salten y después vuelven a autoajustarse.
// Las transiciones posteriores (cambio de variable, agrupación o tema) las
// hace dcc.Graph con animate=True.
(function () {
  // Con movimiento reducido, el CSS muestra los gráficos sin animación
  if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  const DURACION = 700;
  const animados = new WeakSet();

  // El gráfico permanece invisible (CSS) hasta que sus barras están en cero
  function mostrar(gd) { gd.classList.add("grafico-listo"); }

  function esAnimable(t) {
    return t.type === "bar" || (t.type === "scatter" && String(t.mode || "lines").includes("lines"));
  }

  function crecer(gd) {
    if (animados.has(gd) || !gd._fullData || !gd._fullData.length || !window.Plotly) return;
    const indices = [];
    const ceros = [];
    const reales = [];
    // _fullData guarda los valores ya decodificados (Plotly 6 envía los
    // arreglos numéricos en binario: {dtype, bdata}).
    (gd._fullData || []).forEach(function (t) {
      const i = t.index;
      if (!esAnimable(t)) return;
      const eje = t.type === "bar" && t.orientation === "h" ? "x" : "y";
      const valores = t[eje];
      if (!valores || !valores.length) return;
      indices.push(i);
      ceros.push({ [eje]: Array.from(valores, function () { return 0; }) });
      reales.push({ [eje]: Array.from(valores) });
    });
    animados.add(gd);
    if (!indices.length) { mostrar(gd); return; }

    const fl = gd._fullLayout || {};
    const fijar = {};
    const liberar = {};
    ["xaxis", "yaxis"].forEach(function (eje) {
      const explicito = gd.layout && gd.layout[eje] && gd.layout[eje].range;
      if (fl[eje] && fl[eje].range && !explicito) {
        fijar[eje + ".range"] = fl[eje].range.slice();
        fijar[eje + ".autorange"] = false;
        liberar[eje + ".autorange"] = true;
      }
    });

    const instantaneo = { transition: { duration: 0 }, frame: { duration: 0, redraw: false } };
    const suave = { transition: { duration: DURACION, easing: "cubic-out" },
                    frame: { duration: DURACION, redraw: false } };
    Plotly.relayout(gd, fijar)
      .then(function () { return Plotly.animate(gd, { data: ceros, traces: indices }, instantaneo); })
      .then(function () { mostrar(gd); return Plotly.animate(gd, { data: reales, traces: indices }, suave); })
      .then(function () { return Plotly.relayout(gd, liberar); })
      .catch(function () { mostrar(gd); /* si Dash redibuja a mitad de la animación, se ignora */ });
  }

  // Fundido al redibujar con Plotly.react un gráfico ya visible (cajas, donas,
  // heatmaps o cambios en el número de trazas, que Plotly no puede animar).
  function parchearReact() {
    if (!window.Plotly || window.Plotly.__conFundido) return;
    const original = window.Plotly.react;
    window.Plotly.react = function (gd) {
      const args = arguments;
      const el = typeof gd === "string" ? document.getElementById(gd) : gd;
      if (!el || !el.classList || !el.classList.contains("grafico-listo")) {
        return original.apply(window.Plotly, args);
      }
      el.classList.add("grafico-actualizando");
      return new Promise(function (r) { setTimeout(r, 160); })
        .then(function () { return original.apply(window.Plotly, args); })
        .then(function (res) { el.classList.remove("grafico-actualizando"); return res; },
              function (err) { el.classList.remove("grafico-actualizando"); throw err; });
    };
    window.Plotly.__conFundido = true;
  }

  // Revisa los gráficos nuevos tras cada cambio del DOM (con un pequeño
  // retardo para no interferir mientras Dash termina de dibujar).
  let pendiente = null;
  function revisar() {
    clearTimeout(pendiente);
    pendiente = setTimeout(function () {
      parchearReact();
      document.querySelectorAll(".js-plotly-plot").forEach(function (gd) {
        if (gd._fullLayout && gd._fullData && gd._fullData.length) crecer(gd);
      });
    }, 30);
  }

  new MutationObserver(revisar).observe(document.body, { childList: true, subtree: true });
})();
