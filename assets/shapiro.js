// Guía local del dashboard. La base editorial y la conversación viven en el navegador.
(function () {
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
      tema: /\b(mann[\s-]+whitney|chi[\s-]+cuadrado|analisis por sexo|prevalencia por sexo|diferencias? (?:entre|por) (?:chicos|chicas|sexo))\b/,
      texto: "EDA → Análisis por sexo usa chi-cuadrado para evaluar la asociación entre sexo y la variable objetivo binaria. Mann–Whitney compara los rangos de las variables numéricas entre los grupos.",
      enlace: ["analisis_sexo", "Ir a Análisis por sexo"],
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
      tema: /\b(benchmark\w*|validacion cruzada|cv|conjunto de prueba|compar\w*.*modelos?)\b/,
      texto: "En Modelado → Benchmarking puedes comparar clasificación y regresión. La validación cruzada divide el conjunto de entrenamiento en varios pliegues y la evaluación final usa un conjunto de prueba reservado.",
      enlace: ["benchmarking", "Ir a Benchmarking"],
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

  const consultaPersonal = /\b(suicid\w*|autolesion\w*|matarme|quitarme la vida|tengo depresion|(?:tengo|tiene|padezco) (?:ansiedad|depresion)|(?:mi|mis) (?:sintomas|diagnostico|medicamentos)|diagnostica(?:me|rme)|que (?:medicamento|tratamiento) (?:tomo|necesito))\b/;

  function responder(pregunta) {
    const texto = pregunta.toLowerCase().normalize("NFD").replace(/\p{M}/gu, "");
    if (consultaPersonal.test(texto)) {
      return {texto: "Para una consulta de salud personal, busca orientación de un profesional de salud. Puedo ayudarte con la navegación y la metodología del dashboard."};
    }
    return guia.find(({tema}) => tema.test(texto)) || {
      texto: "Puedo orientarte sobre las secciones, los gráficos y los métodos del dashboard. Prueba con «¿Qué significa BDI-II?» o «¿Dónde veo los modelos?». Las cifras y los resultados están en las secciones correspondientes.",
    };
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

  function enviar(dialogo, texto, desdeSugerencia = false) {
    const pregunta = texto.trim();
    if (!pregunta) return;

    const ayuda = responder(pregunta);
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

    dialogo.querySelector("#mensajes-shapiro").append(mensaje("Tú", pregunta, "usuario"), respuesta);
    dialogo.querySelector("#sugerencias-shapiro").hidden = true;
    const conversacion = dialogo.querySelector("#conversacion-shapiro");
    conversacion.scrollTop = conversacion.scrollHeight;
    const campo = dialogo.querySelector("#pregunta-shapiro");
    campo.value = "";
    if (desdeSugerencia) {
      respuesta.tabIndex = -1;
      respuesta.focus({preventScroll: true});
    } else {
      campo.focus({preventScroll: true});
    }
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
