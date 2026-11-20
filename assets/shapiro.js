// Conversación de Shapiro y guía local de respaldo.
(function () {
  let historial = [];
  let esperando = false;
  const guia = [
    {
      tema: /\b(umbral (?:de |del )?(?:modelo|clasificacion|probabilidad|tamizaje)|umbral.*catboost)\b/,
      texto: "El umbral del modelo es la probabilidad a partir de la cual CatBoost marca un perfil como positivo en el tamizaje. El corte del BDI-II, en cambio, se aplica al puntaje del cuestionario para definir la variable objetivo.",
      enlace: ["modelos_finales", "Ver modelos finales"],
    },
    {
      tema: /\b(prediccion\w*|simulador|simula\w*|calcular un perfil|diagnostico)\b/,
      texto: "En Modelado → Predicción puedes ajustar un perfil de ejemplo y pulsar Calcular. CatBoost estima la probabilidad para el objetivo binario y ElasticNet estima el puntaje BDI-II. Los resultados se usan con fines académicos.",
      enlace: ["prediccion", "Ir a Predicción"],
    },
    {
      tema: /\b(?:chi|ji)(?:[\s-]+(?:al[\s-]+)?cuadrad[oa]|[\s^]*2)\b|\bchi²|χ(?:²|\^?2)/,
      texto: "El test chi-cuadrado de independencia (χ²) compara los conteos observados con los que esperaríamos si dos variables categóricas fueran independientes. Aquí cruza el sexo con la clasificación del BDI-II. El proyecto usa p < 0,05 como criterio de evidencia de asociación. Para entender su magnitud, mira la V de Cramér. Para ver qué grupo tiene mayor prevalencia, compara los porcentajes.",
      enlace: ["analisis_sexo", "Ir a Análisis por sexo"],
    },
    {
      tema: /\b(?:estado depresivo|prevalencia|depresion|sintomas depresivos|bdi[\s_-]*ii)\b.*\b(?:sexo|genero|chicas|chicos)\b|\b(?:analisis por sexo|diferencias? (?:entre|por) (?:chicos|chicas|sexo))\b/,
      texto: "El gráfico Estado depresivo por sexo muestra, para chicas y chicos, el porcentaje con 14 puntos o más en el BDI-II y el porcentaje por debajo de ese corte. Cada porcentaje se calcula sobre el total de su propio grupo. Para comparar la prevalencia, mira el segmento que alcanzó el corte en cada grupo. La tabla de chi-cuadrado te ayuda a evaluar si hay evidencia estadística de una asociación.",
      enlace: ["analisis_sexo", "Ir a Análisis por sexo"],
    },
    {
      tema: /\bmann[\s-]+whitney\b/,
      texto: "La prueba U de Mann–Whitney reúne y ordena los valores de ambos grupos para evaluar si los de uno tienden a ser más altos que los del otro. En Análisis por sexo se usa para variables numéricas. Junto con el valor p, el dashboard muestra el tamaño del efecto y las medianas de cada grupo.",
      enlace: ["analisis_sexo", "Ir a Análisis por sexo"],
    },
    {
      tema: /\b(benchmark\w*|validacion cruzada|cv|conjunto de prueba|compar\w*.*\bmodelos?(?! por (?:sexo|genero))|mejor(?:es)? (?:modelos?|algoritmos?)(?! por (?:sexo|genero))|(?:modelos?|algoritmos?) (?:ganador(?:es)?|.*\b(?:gano|ganaron|mejor(?:es)?))|mayor pr[\s-]*auc|menor rmse)\b/,
      texto: [
        "En Modelado → Benchmarking, elige una opción en Tarea y pulsa Mostrar todos. La tabla muestra la media ± desviación estándar entre los pliegues de validación. El rombo del gráfico marca el modelo final.",
        "Clasificación · depressed predice la clase binaria. Busca la PR-AUC más alta y revisa las curvas ROC y Precision-Recall sobre el mismo conjunto de prueba. La regresión logística ponderada es el modelo de referencia. Su fila de validación cruzada viene del EDA, con otra partición. La prevalencia marca el nivel del azar en PR-AUC.",
        "Regresión · bdi_total predice el puntaje BDI-II. Busca el RMSE más bajo, en puntos del BDI-II, y compáralo con la regresión lineal de referencia. En Modelado → Modelos finales puedes consultar las métricas de prueba del modelo final.",
      ].join("\n\n"),
      enlace: ["benchmarking", "Ir a Benchmarking"],
    },
    {
      tema: /\b(bdi(?:[\s_-]*ii)?|bdi_total|depressed|beck|variable objetivo|punto de corte|umbral de 14)\b/,
      texto: "BDI-II es el Inventario de Depresión de Beck, segunda edición. El puntaje resume la intensidad de los síntomas depresivos. En este estudio, un puntaje de 14 o más define la clase positiva para el análisis.",
      enlace: ["variable_objetivo", "Ir a Variable objetivo"],
    },
    {
      tema: /\b(correlac\w*|spearman|causalidad|mapa de calor|heatmap)\b/,
      texto: "En EDA → Correlaciones, Spearman mide la asociación entre los rangos de dos variables. Un signo positivo indica que tienden a subir juntas y un signo negativo que una tiende a bajar cuando la otra sube. La correlación por sí sola es insuficiente para establecer causalidad.",
      enlace: ["correlaciones", "Ir a Correlaciones"],
    },
    {
      tema: /\b(outlier\w*|atipic\w*|iqr|rango intercuartil\w*)\b/,
      texto: "En EDA → Outliers, se señalan los valores que caen por debajo de Q1 − 1,5 × RIQ o por encima de Q3 + 1,5 × RIQ. Q1 y Q3 son los cuartiles y RIQ es su diferencia.",
      enlace: ["outliers", "Ir a Outliers"],
    },
    {
      tema: /\b(modelos? por (?:sexo|genero)|genero|importancia por (?:sexo|grupo))\b/,
      texto: "En Modelado → Modelos por género puedes consultar por separado la evaluación y la importancia de las variables para chicos y chicas.",
      enlace: ["genero", "Ver modelos por género"],
    },
    {
      tema: /\b(histograma\w*|distribuc\w*|boxplot\w*|diagrama de caja\w*|variable[s]? numerica[s]?|filtro\w*|filtrar|separar por sexo)\b/,
      texto: "En EDA → Variables numéricas, elige una variable y, si te interesa, un grupo. El histograma muestra la frecuencia por intervalos y el diagrama de caja resume la mediana, la dispersión y los valores alejados.",
      enlace: ["analisis_numerico", "Ir a Variables numéricas"],
    },
    {
      tema: /\b(nulos|duplicados|calidad de datos|valores faltantes|dimensiones)\b/,
      texto: "EDA → Calidad de datos reúne las dimensiones del dataset y los controles de valores faltantes, duplicados y rangos de las variables.",
      enlace: ["calidad_datos", "Ir a Calidad de datos"],
    },
    {
      tema: /\b(cluster\w*|k[\s-]*means|silueta|agrupamiento)\b/,
      texto: "Modelado → Clustering usa K-Means para describir perfiles con hábitos de pantalla y sueño similares. La sección explica la elección del número de grupos y muestra cómo se caracterizan.",
      enlace: ["clustering", "Ir a Clustering"],
    },
    {
      tema: /\b(modelado|modelos?|catboost|elastic[\s-]*net|roc|auc|rmse|regresion|clasificacion)\b/,
      texto: "Modelado → Modelos finales documenta CatBoost para la clasificación y ElasticNet para el puntaje BDI-II. Allí puedes consultar las métricas, la importancia de las variables y la elección del umbral del clasificador.",
      enlace: ["modelos_finales", "Ver modelos finales"],
    },
    {
      tema: /\b(metod\w*|preproces\w*|variables del (?:estudio|dataset)|como se (?:hizo|realizo|analizo))\b/,
      texto: "Proyecto → Metodología documenta las variables, el análisis exploratorio, el preprocesamiento, el modelado y la evaluación. El Marco teórico reúne los antecedentes y la base estadística.",
      enlace: ["metodologia", "Leer la metodología"],
    },
    {
      tema: /\b(objetivos? (?:del |de este )?(?:estudio|proyecto)|objetivo general|objetivos especificos)\b/,
      texto: "Proyecto → Objetivos presenta la pregunta general del estudio y los objetivos específicos sobre pantallas, sueño y síntomas depresivos.",
      enlace: ["objetivos", "Ver los objetivos"],
    },
    {
      tema: /\b(marco teorico|antecedentes|base estadistica)\b/,
      texto: "Proyecto → Marco teórico reúne los antecedentes del estudio y la base estadística para interpretar el análisis.",
      enlace: ["marco_teorico", "Ir al Marco teórico"],
    },
    {
      tema: /\b(kaggle|dataset|csv|fuente\w*|descarg\w*.*datos|autores|equipo|jupyter|github|repositorio)\b/,
      texto: "En Acerca de → Equipo y recursos están los enlaces al Jupyter Book, al repositorio y al dataset publicado en Kaggle.",
      enlace: ["acerca", "Ver equipo y recursos"],
    },
    {
      tema: /\b(estudio|muestra|participantes|adolescentes|prevalencia|resumen|empezar|inicio)\b/,
      texto: "Proyecto → Resumen del estudio presenta el contexto, los indicadores generales de la muestra y el índice de secciones del dashboard.",
      enlace: ["inicio", "Ir al resumen del estudio"],
    },
    {
      tema: /\b(eda|exploratorio)\b/,
      texto: "EDA significa análisis exploratorio de datos. En ese menú puedes revisar la calidad de los datos, las distribuciones, las diferencias entre grupos, las correlaciones y los valores atípicos.",
      enlace: ["calidad_datos", "Empezar por Calidad de datos"],
    },
    {
      tema: /\b(portada|pagina principal)\b/,
      texto: "Puedes volver a la portada con el enlace «Pantallas y bienestar» de la cabecera.",
      enlace: ["", "Volver a la portada"],
    },
    {
      tema: /\b(tema|modo (?:oscuro|claro)|pantalla (?:oscura|clara))\b/,
      texto: "Usa los iconos de sol y luna en la cabecera para cambiar de tema. La preferencia se guarda en el navegador y se aplica también a los gráficos.",
    },
    {
      tema: /\b(grafic\w*|zoom|leyenda|acercar)\b/,
      texto: "Pasa el cursor o toca una marca del gráfico para ver su detalle. Si el gráfico permite zoom, arrastra para seleccionar una zona y haz doble clic para restablecer la vista. En pantallas estrechas, algunas tablas y gráficos se desplazan horizontalmente.",
    },
    {
      tema: /\b(navega\w*|menu|secciones|dashboard|shapiro|que puedes hacer)\b/,
      texto: "Abre Proyecto, EDA, Modelado o Acerca de desde el menú superior. El Resumen del estudio también incluye enlaces directos a las secciones.",
      enlace: ["inicio", "Abrir el índice de secciones"],
    },
  ];

  const consultaPersonal = /\b(suicid\w*|autolesion\w*|matarme|quitarme la vida|(?:tengo|padezco) (?:ansiedad|depresion)|(?:mi|mis) (?:sintomas|diagnostico|medicamentos)|diagnostica(?:me|rme)|que (?:medicamento|tratamiento) (?:tomo|necesito))\b/;

  function responderLocal(pregunta) {
    const texto = pregunta.toLowerCase().normalize("NFD").replace(/\p{M}/gu, "");
    if (consultaPersonal.test(texto)) {
      return {texto: "Para una consulta de salud personal, busca orientación de un profesional de salud. Puedo ayudarte con la navegación y la metodología del dashboard."};
    }
    return guia.find(({tema}) => tema.test(texto)) || {
      texto: "Puedo orientarte sobre las secciones, los gráficos y los métodos del dashboard. Prueba con «¿Qué significa BDI-II?» o «¿Dónde veo los modelos?». Las cifras y los resultados están en las secciones correspondientes.",
    };
  }

  async function responder(dialogo, pregunta) {
    const texto = pregunta.toLowerCase().normalize("NFD").replace(/\p{M}/gu, "");
    if (!dialogo.dataset.api || consultaPersonal.test(texto)) return responderLocal(pregunta);

    const mensajes = [...historial, {role: "user", content: pregunta}];
    let aviso = "Gemini no está disponible en este momento.";
    try {
      const url = new URL(dialogo.dataset.api, window.location.origin);
      const ruta = window.location.pathname;
      const raiz = dialogo.dataset.raiz;
      if (ruta.startsWith(raiz)) {
        const seccion = ruta.slice(raiz.length).replace(/\/+$/, "");
        url.searchParams.set("seccion", seccion || "portada");
        const filtros = seccion === "analisis_numerico"
          ? {variable: "#num-variable", agrupacion: "#num-separar input:checked"}
          : seccion === "benchmarking" ? {tarea: "#bm-tarea input:checked"} : {};
        for (const [parametro, selector] of Object.entries(filtros)) {
          const valor = document.querySelector(selector)?.value;
          if (valor) url.searchParams.set(parametro, valor);
        }
      }
      const respuesta = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(mensajes),
        signal: AbortSignal.timeout(33000),
      });
      if (respuesta.status === 429) aviso = "Se alcanzó un límite de consultas.";
      const ayuda = await respuesta.json();
      if (!respuesta.ok) {
        if (typeof ayuda.error === "string") aviso = ayuda.error;
        throw new Error(aviso);
      }
      historial = [...mensajes, {role: "assistant", content: ayuda.texto}]
        .slice(-2 * Number(dialogo.dataset.intercambios));
      return ayuda;
    } catch {
      const ayuda = responderLocal(pregunta);
      return {...ayuda, texto: aviso + " Guía local.\n\n" + ayuda.texto};
    }
  }

  function mensaje(autor, texto, clase) {
    const bloque = document.createElement("div");
    bloque.className = `mensaje-shapiro mensaje-shapiro--${clase}`;
    const etiqueta = document.createElement("span");
    etiqueta.className = "visualmente-oculto";
    etiqueta.textContent = autor;
    const cuerpo = document.createElement("p");
    cuerpo.textContent = texto;
    bloque.append(etiqueta, cuerpo);
    return bloque;
  }

  async function enviar(dialogo, texto, desdeSugerencia = false) {
    const pregunta = texto.trim();
    if (!pregunta || esperando) return;

    const mensajes = dialogo.querySelector("#mensajes-shapiro");
    const sugerencias = dialogo.querySelector("#sugerencias-shapiro");
    const conversacion = dialogo.querySelector("#conversacion-shapiro");
    const campo = dialogo.querySelector("#pregunta-shapiro");
    const boton = dialogo.querySelector(".shapiro-enviar");
    const estado = dialogo.querySelector("#estado-shapiro");
    mensajes.append(mensaje("Tú", pregunta, "usuario"));
    campo.value = "";
    if (desdeSugerencia) campo.focus({preventScroll: true});
    sugerencias.hidden = true;
    esperando = true;
    boton.disabled = true;
    estado.textContent = dialogo.dataset.api ? "Shapiro está respondiendo…" : "";
    conversacion.scrollTop = conversacion.scrollHeight;
    const ayuda = await responder(dialogo, pregunta);
    esperando = false;
    boton.disabled = false;
    estado.textContent = "";
    const respuesta = mensaje("Shapiro", ayuda.texto, "guia");
    if (ayuda.enlace) {
      const [seccion, etiqueta] = ayuda.enlace;
      const enlace = document.createElement("a");
      enlace.href = `${dialogo.dataset.raiz}${seccion}`;
      enlace.textContent = etiqueta;
      enlace.className = "shapiro-enlace";
      const flecha = document.createElement("span");
      flecha.setAttribute("aria-hidden", "true");
      flecha.textContent = " →";
      enlace.append(flecha);
      enlace.addEventListener("click", (evento) => {
        if (evento.metaKey || evento.ctrlKey || evento.shiftKey || evento.altKey) return;
        if (window.dash_clientside?.set_props) {
          evento.preventDefault();
          dialogo.close();
          window.dash_clientside.set_props("url", {pathname: enlace.pathname, search: "", hash: ""});
          window.scrollTo(0, 0);
        }
      });
      respuesta.append(enlace);
    }

    mensajes.append(respuesta);
    conversacion.scrollTop = conversacion.scrollHeight;
  }

  document.addEventListener("click", (evento) => {
    const dialogo = document.getElementById("shapiro");
    if (!dialogo) return;
    if (evento.target.closest("#abrir-shapiro")) {
      dialogo.showModal();
      if (!window.matchMedia("(pointer: coarse)").matches) dialogo.querySelector("#pregunta-shapiro").focus();
    } else if (evento.target.closest("#cerrar-shapiro") || evento.target === dialogo) {
      dialogo.close();
    } else if (evento.target.matches(".shapiro-sugerencia")) {
      enviar(dialogo, evento.target.textContent, true);
    }
  });

  document.addEventListener("submit", (evento) => {
    if (evento.target.id !== "formulario-shapiro") return;
    evento.preventDefault();
    enviar(evento.target.closest("dialog"), evento.target.querySelector("#pregunta-shapiro").value);
  });

  document.addEventListener("keydown", (evento) => {
    if (evento.target.id !== "pregunta-shapiro" || evento.key !== "Enter" || evento.shiftKey || evento.isComposing) return;
    evento.preventDefault();
    evento.target.form.requestSubmit();
  });
})();
