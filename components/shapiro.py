"""Interfaz de Shapiro para la guía local y la conexión a Gemini."""

from dash import get_relative_path, html

PREGUNTAS = (
    "¿Qué significa BDI-II?",
    "¿Cómo leo una correlación?",
    "¿Qué es el test de chi-cuadrado?",
    "¿Qué modelos se eligieron y por qué?",
)


def boton_shapiro():
    return html.Button([
        html.Span(className="icono-chat", **{"aria-hidden": "true"}), "Shapiro",
    ], type="button", id="abrir-shapiro", className="boton-shapiro", disable_n_clicks=True,
        **{"aria-haspopup": "dialog", "aria-controls": "shapiro"})


def dialogo_shapiro(gemini_activo: bool):
    privacidad = ("Gemini procesa tus preguntas, el historial reciente y el contexto del dashboard. "
                  "Evita incluir información personal."
                  if gemini_activo else "Las preguntas se procesan en este navegador.")
    return html.Dialog([
        html.Header([
            html.Span(html.Span(className="icono-chat"), className="shapiro-marca", **{"aria-hidden": "true"}),
            html.Div([
                html.H2("Shapiro", id="titulo-shapiro"),
                html.P("Guía del dashboard"),
            ], className="shapiro-titulo"),
            html.Button("×", type="button", id="cerrar-shapiro", className="shapiro-cerrar",
                        disable_n_clicks=True, **{"aria-label": "Cerrar Shapiro"}),
        ], className="shapiro-cabecera"),
        html.Div([
            html.Div(
                html.P("Hola, soy Shapiro. Pregúntame cómo navegar el dashboard o interpretar sus gráficos y métodos.",
                       className="mensaje-shapiro mensaje-shapiro--guia"),
                id="mensajes-shapiro", role="log", **{"aria-label": "Conversación con Shapiro"},
            ),
            html.Div([
                html.Button(pregunta, type="button", className="shapiro-sugerencia", disable_n_clicks=True)
                for pregunta in PREGUNTAS
            ], id="sugerencias-shapiro"),
            html.P(id="estado-shapiro", className="shapiro-estado", role="status"),
        ], className="shapiro-conversacion", id="conversacion-shapiro"),
        html.Form([
            html.Label("Escribe tu pregunta para Shapiro", htmlFor="pregunta-shapiro", className="visualmente-oculto"),
            html.Div([
                html.Textarea(id="pregunta-shapiro", rows=1, placeholder="Escribe tu pregunta…", maxLength=280,
                              autoComplete="off", required=True),
                html.Button("Enviar", type="submit", className="shapiro-enviar", disable_n_clicks=True),
            ], className="shapiro-campo"),
            html.P(privacidad, className="shapiro-privacidad"),
        ], id="formulario-shapiro"),
    ], id="shapiro", className="shapiro", **{
        "aria-labelledby": "titulo-shapiro", "data-raiz": get_relative_path("/"),
        "data-api": get_relative_path("/api/shapiro") if gemini_activo else "",
    })
