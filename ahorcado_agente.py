# --- Imports ---
from collections import Counter, defaultdict
import unicodedata, re, random, math, os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# ==============================
#  Normalización y carga de corpus
# ==============================

def normalize(text, keep_accents=False):
    t = text.strip().lower()
    if not keep_accents:
        t = ''.join(c for c in unicodedata.normalize('NFD', t)
                    if unicodedata.category(c) != 'Mn')
    t = re.sub(r'[^a-zñ]', '', t)
    return t

def load_corpus(path, keep_accents=False, min_len=4, max_len=12):
    with open(path, 'r', encoding='utf-8') as f:
        words = [normalize(w, keep_accents) for w in f]
    return [w for w in words if min_len <= len(w) <= max_len]

def ensure_corpus(path):
    """
    Si no existe el archivo de corpus, crea uno sencillo de ejemplo.
    """
    if os.path.exists(path):
        print(f"✅ Encontrado corpus: {path}")
        return path

    sample_words = """
    hola adios casa perro gato mesa silla libro escuela juego trabajo examen codigo algoritmo
    dato modelo matriz vector funcion programa variable constante entero flotante cadena lista
    arbol grafo busqueda camino heuristica inteligencia agente estado accion resultado problema
    ciencia computacion aprendizaje maquina redes internet servidor cliente sistema modulo clase
    objeto metodo interfaz diseño prueba error depuracion rendimiento tiempo espacio complejidad
    estrategia frecuencia posicion patron letra palabra frase oracion parrafo idioma español ingles
    mexico chile argentina peru colombia españa francia alemania italia canada estadosunidos brasil
    futbol portero defensa medio delantero gol estadio aficion mundial balon liga arbitro jugador
    clima calor frio lluvia viento nublado soleado nevado verano invierno otoño primavera ciudad
    coche avion barco tren viaje ruta mapa calle avenida plaza parque museo playa montaña rio lago
    salud doctor hospital medicina vacuna ejercicio fuerza cardio descanso sueño alimentacion proteina
    energia bateria circuito sensor robot motor torque presion voltaje corriente resistencia
    pantalla teclado raton monitor laptop telefono tablet reloj auricular microfono camara video
    musica ritmo melodia armonia nota guitarra piano bateria bajo canto voz concierto festival
    arte pintura escultura fotografia lienzo color sombra luz perspectiva composicion contraste
    historia cultura filosofia etica estetica logica semantica sintaxis morfologia fonetica
    matematicas algebra calculo geometria probabilidad estadistica combinatoria optimizacion
    economia mercado oferta demanda precio costo gasto ingreso beneficio inversion ahorro
    psicologia emocion atencion memoria aprendizaje conducta percepcion lenguaje pensamiento
    deporte correr nadar saltar pesas gimnasio maraton yoga box futbol basquet tenis voleibol
    comida pan queso leche huevo arroz frijol tortilla sopa salsa taco burrito enchilada pozole
    postre pastel galleta chocolate helado flan gelatina fruta manzana pera uva platano naranja
    tecnologia software hardware dato nube colab github repositorio version control rama commit
    merge pull request issue release licencia documentacion prueba automatica integracion despliegue
    seguridad clave cifrado token autenticacion autorizacion permiso rol usuario sesion cookie
    interfaz boton lista tabla grafica filtro busqueda orden sort paginacion responsive accesibilidad
    """
    sample_words = [w for w in sample_words.split() if w]
    with open(path, 'w', encoding='utf-8') as f:
        for w in sample_words:
            f.write(w + "\n")
    print(f"📄 No se encontró '{path}'. Se creó un corpus de ejemplo con {len(sample_words)} palabras.")
    return path

# ==============================
#  Filtrado de candidatos
# ==============================

def filter_candidates(words, pattern, excluded, included_positions):
    rx = '^' + pattern.replace('_', '.') + '$'
    r = re.compile(rx)
    res = []
    for w in words:
        if not r.match(w):
            continue
        if any(ch in w for ch in excluded):
            continue
        ok = True
        for i, ch in included_positions.items():
            if i >= len(w) or w[i] != ch:
                ok = False
                break
        if ok:
            res.append(w)
    return res

# ==============================
#  Heurísticas
# ==============================

def pick_by_conditional_freq(cands, tried):
    cnt = Counter()
    for w in cands:
        for ch in set(w):
            if ch not in tried:
                cnt[ch] += 1
    return max(cnt, key=cnt.get) if cnt else None

def pick_by_info_gain(cands, tried):
    N = len(cands)
    if N <= 1:
        return None
    best, best_gain = None, -1
    for ch in 'abcdefghijklmnñopqrstuvwxyz':
        if ch in tried:
            continue
        buckets = {}
        for w in cands:
            mask = tuple(i for i, c in enumerate(w) if c == ch)  # () si falla
            buckets[mask] = buckets.get(mask, 0) + 1
        H = 0.0
        for k in buckets.values():
            p = k / N
            H += -p * math.log2(max(p, 1e-12))
        if H > best_gain:
            best_gain, best = H, ch
    return best

def pick_by_lookahead(
    cands, tried, pattern, included_positions,
    errors_left=None, max_errors=None,
    alpha=1.0, beta=1.5, gamma=0.15
):
    """
    1-step lookahead:
      score(ch) = sum_buckets p(b) * [ alpha*(-log2(|C_b|+1))
                                       - beta*I(b=miss)
                                       + gamma*len(mask_b) ]
    """
    if not cands:
        return None
    N = len(cands)
    alphabet = 'abcdefghijklmnñopqrstuvwxyz'
    best, best_score = None, -1e18

    for ch in alphabet:
        if ch in tried:
            continue

        buckets = defaultdict(list)  # mask -> lista de palabras
        for w in cands:
            mask = tuple(i for i, c in enumerate(w) if c == ch)  # () = fallo
            buckets[mask].append(w)

        score = 0.0
        for mask, group in buckets.items():
            p = len(group) / N
            size_term = -math.log2(len(group) + 1)
            miss_penalty = 0.0
            reveal_bonus = gamma * len(mask)

            if len(mask) == 0:  # fallo
                if errors_left is not None and errors_left <= 0:
                    miss_penalty = 1e6  # súper penalización si ya no hay vidas
                else:
                    miss_penalty = beta

            branch_utility = alpha * size_term - miss_penalty + reveal_bonus
            score += p * branch_utility

        if score > best_score:
            best_score, best = score, ch

    return best

# ==============================
#  Agente
# ==============================

class HangmanAgent:
    def __init__(self, words, scorer="lookahead"):
        self.words = words
        self.scorer = scorer

    def next_guess(self, pattern, excluded, included_positions, tried,
                   errors_left=None, max_errors=None):
        cands = filter_candidates(self.words, pattern, excluded, included_positions)
        if not cands:
            return None, []
        if self.scorer == "freq":
            ch = pick_by_conditional_freq(cands, tried)
        elif self.scorer == "info":
            ch = pick_by_info_gain(cands, tried)
        else:  # "lookahead"
            ch = pick_by_lookahead(
                cands, tried, pattern, included_positions,
                errors_left, max_errors
            )
        return ch, cands

# ==============================
#  Simulación (texto, por si la quieres usar)
# ==============================

def play(word, agent, max_errors=6):
    pattern = '_' * len(word)
    excluded, tried, included = set(), set(), {}
    errors = 0
    while '_' in pattern and errors < max_errors:
        ch, _ = agent.next_guess(pattern, excluded, included, tried,
                                 errors_left=max_errors-1-errors, max_errors=max_errors)
        if ch is None:
            break
        tried.add(ch)
        if ch in word:
            pattern = ''.join(ch if word[i] == ch else pattern[i] for i in range(len(word)))
            for i, c in enumerate(word):
                if c == ch:
                    included[i] = c
        else:
            excluded.add(ch)
            errors += 1
    return (pattern == word), errors

# ==============================
#  Dificultad / selección de palabra
# ==============================

def build_letter_freq(words):
    total, freq = 0, {ch: 0 for ch in "abcdefghijklmnñopqrstuvwxyz"}
    for w in words:
        for ch in w:
            if ch in freq:
                freq[ch] += 1
                total += 1
    for ch in freq:
        freq[ch] = freq[ch] / max(total, 1)
    return freq

def word_rarity(word, freq):
    vals = [(1.0 - freq.get(ch, 0.0)) for ch in word if ch in freq]
    return sum(vals) / len(vals) if vals else 0.0

def pick_word_by_difficulty(pool_words, freq, difficulty):
    words_scored = [(w, word_rarity(w, freq)) for w in pool_words]
    words_scored.sort(key=lambda x: x[1])
    n = len(words_scored)

    if difficulty == "facil":
        candidates = words_scored[: max(1, int(0.40 * n))]
        candidates = [p for p in candidates if 5 <= len(p[0]) <= 8] or candidates
    elif difficulty == "medio":
        candidates = words_scored
    elif difficulty == "dificil":
        candidates = words_scored[max(0, int(0.70 * n)):]
        candidates = [p for p in candidates if len(p[0]) >= 7] or candidates
    else:  # "imposible" u otro
        candidates = words_scored[max(0, int(0.85 * n)):]
        candidates = [p for p in candidates if len(p[0]) >= 7] or candidates

    return random.choice(candidates)[0]

def make_noisy_agent(words, scorer="lookahead", noise=0.0):
    """
    Envuelve a HangmanAgent con algo de 'ruido':
    - Con probabilidad = noise, elige una letra aleatoria.
    - Si no, usa la heurística normal del agente.
    """
    base = HangmanAgent(words, scorer=scorer)
    alphabet = list("abcdefghijklmnñopqrstuvwxyz")

    # Guardamos la versión original ANTES de sobrescribirla
    original_next = base.next_guess

    def next_guess_noisy(pattern, excluded, included_positions, tried,
                         errors_left=None, max_errors=None):
        # Ruido: a veces elige una letra random
        if random.random() < noise:
            choices = [c for c in alphabet if c not in tried and c not in excluded]
            ch = random.choice(choices) if choices else None
            return ch, []

        # Si no hay ruido, usamos la función original
        return original_next(pattern, excluded, included_positions, tried,
                             errors_left, max_errors)

    base.next_guess = next_guess_noisy
    return base

LEVELS = {
    # Más vidas y cero ruido → realmente fácil
    "facil":      {"max_errors": 10, "noise": 0.00, "scorer": "lookahead"},
    "medio":      {"max_errors": 6,  "noise": 0.03, "scorer": "lookahead"},
    "dificil":    {"max_errors": 5,  "noise": 0.10, "scorer": "lookahead"},
    "imposible":  {"max_errors": 4,  "noise": 0.18, "scorer": "lookahead"},
}

# ==============================
#  Gráficos del ahorcado (Matplotlib puro)
# ==============================

def draw_hangman_matplotlib_spacious(ax, errors, pattern, tried, max_errors=6):
    """
    Dibuja el ahorcado en función del número de errores.
    Escala automáticamente para que el mono SIEMPRE se dibuje completo
    cuando se llegue al máximo de errores, aunque max_errors no sea 6.
    """
    ax.clear()
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 22)
    ax.axis('off')

    # ======== MAPEO DE ERRORES → ETAPAS DE DIBUJO (0..6) ========
    steps = 6  # cabeza + cuerpo + 2 brazos + 2 piernas
    if max_errors > 0:
        stage = int(round(min(errors, max_errors) / max_errors * steps))
    else:
        stage = errors

    # ======== ESTRUCTURA DEL AHORCADO ========
    ax.plot([3, 13], [4, 4], color="#444", linewidth=4)      # base
    ax.plot([4, 4], [4, 20], color="#444", linewidth=4)      # poste
    ax.plot([4, 10], [20, 20], color="#444", linewidth=4)    # viga
    ax.plot([10, 10], [20, 18.6], color="#444", linewidth=3) # cuerda

    # ======== DIBUJO DEL MUÑECO ========
    if stage >= 1:
        head = plt.Circle((10, 17.6), 1.0, fill=False, color="#000", linewidth=3)
        ax.add_patch(head)
    if stage >= 2:
        ax.plot([10, 10], [16.6, 13.8], color="#3366ff", linewidth=3)  # cuerpo
    if stage >= 3:
        ax.plot([10, 8.8], [16.1, 15.0], color="#cc3333", linewidth=3) # brazo izq
    if stage >= 4:
        ax.plot([10, 11.2], [16.1, 15.0], color="#cc3333", linewidth=3) # brazo der
    if stage >= 5:
        ax.plot([10, 9.0], [13.8, 11.6], color="#33aa33", linewidth=3) # pierna izq
    if stage >= 6:
        ax.plot([10, 11.0], [13.8, 11.6], color="#33aa33", linewidth=3) # pierna der

    # ======== TEXTO DE PALABRA Y LETRAS INTENTADAS ========
    ax.text(8, 9.2, "Palabra:", fontsize=15, ha='center', va='center', fontweight='bold')

    ax.text(
        8,
        8.0,
        " ".join(list(pattern.upper())),
        fontsize=26,
        family="monospace",
        ha='center',
        va='center',
        color="#000"
    )

    tried_txt = ', '.join(sorted(tried)) if tried else '—'
    ax.text(
        8,
        6.4,
        f"Letras intentadas: {tried_txt}",
        fontsize=12,
        ha='center',
        va='center',
        color="#333"
    )


def play_visual_graphic_spacious(word, agent, max_errors=6, sleep=1.0,
                                 fig=None, ax=None):
    """
    Juega una partida mostrando el ahorcado en una figura de matplotlib.
    - Si fig/ax vienen en None, crea una nueva ventana.
    - Si ya existen, reutiliza la misma (para varias partidas).
    Además, al final muestra el resultado (GANÓ / PERDIÓ) dentro de la figura.
    """
    pattern = '_' * len(word)
    excluded, tried, included = set(), set(), {}
    errors = 0

    # ¿Ya había figura/axes?
    if fig is None or ax is None:
        plt.ion()  # modo interactivo
        fig, ax = plt.subplots(figsize=(7, 10), dpi=120)
        fig.subplots_adjust(left=0.05, right=0.95, top=0.96, bottom=0.18)

    # Bucle del juego (el agente va adivinando solo)
    while '_' in pattern and errors < max_errors:
        draw_hangman_matplotlib_spacious(ax, errors, pattern, tried, max_errors=max_errors)
        fig.canvas.draw()
        plt.pause(sleep)

        ch, _ = agent.next_guess(pattern, excluded, included, tried,
                                 errors_left=max_errors-1-errors,
                                 max_errors=max_errors)
        if ch is None:
            break
        tried.add(ch)
        if ch in word:
            pattern = ''.join(ch if word[i] == ch else pattern[i]
                              for i in range(len(word)))
            for i, c in enumerate(word):
                if c == ch:
                    included[i] = c
        else:
            excluded.add(ch)
            errors += 1

      # ¿Ganó o perdió?
    win = (pattern == word)

    # ----- Ajuste para el dibujo y el contador de errores -----
    # Si perdió (no adivinó la palabra), queremos que:
    # - Se dibuje el mono COMPLETO.
    # - Muestre Errores = max_errors / max_errors, aunque en realidad haya fallado menos
    if win:
        draw_errors = errors
        shown_errors = errors
    else:
        draw_errors = max_errors
        shown_errors = max_errors

    # Estado final del dibujo
    draw_hangman_matplotlib_spacious(ax, draw_errors, pattern, tried, max_errors=max_errors)

    # ===== Texto extra en la propia figura =====
    status = "¡GANÓ! 🎉" if win else "PERDIÓ 😢"
    ax.text(8, 5.0, status,
            fontsize=16, ha='center', va='center',
            color="#000", fontweight='bold')

    resumen = f"Objetivo: {word.upper()}   |   Errores: {shown_errors}/{max_errors}"
    ax.text(8, 3.3, resumen,   # <-- más abajo, para que no lo tape la base
            fontsize=11, ha='center', va='center',
            color="#333")

    fig.canvas.draw()
    plt.pause(2.0)   # pequeña pausa para que se vea claro el resultado

    # También lo dejamos en la terminal por si lo quieres allí
    print("===================================")
    print(f"Palabra objetivo: {word.upper()}")
    print(f"Resultado final:  {pattern.upper()}")
    print(f"Errores totales (mostrados):  {shown_errors} / {max_errors}")
    print("✅ ¡Ganó!" if win else "❌ Perdió")
    print("===================================")

    return win, errors, pattern, fig, ax


# ==============================
#  main con botones de dificultad
# ==============================

def main():
    # Preparar corpus y conjuntos de palabras
    corpus_path = "es_corpus.txt"
    corpus_path = ensure_corpus(corpus_path)
    words = load_corpus(corpus_path, keep_accents=False, min_len=4, max_len=10)
    train, test = words[::2], words[1::2]
    print(f"📚 Corpus listo. Total: {len(words)} | Train: {len(train)} | Test: {len(test)}")

    freq = build_letter_freq(train + test)

    # ----- Crear figura principal -----
    plt.ion()
    fig = plt.figure(figsize=(7, 10), dpi=120)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.96, bottom=0.20)

    # Eje grande para el ahorcado
    ax_main = fig.add_axes([0.05, 0.25, 0.9, 0.7])

    # Dibujo inicial "en blanco"
    draw_hangman_matplotlib_spacious(ax_main, 0, "_______", set(), max_errors=6)
    ax_main.text(8, 3.0, "Haz clic en una dificultad para empezar",
                 fontsize=12, ha='center', va='center', color="#333")
    fig.canvas.draw()

    # ----- Función para iniciar una partida -----
    def start_game(difficulty):
        cfg = LEVELS[difficulty]
        target = pick_word_by_difficulty(test, freq, difficulty)
        agent = make_noisy_agent(train, scorer=cfg["scorer"], noise=cfg["noise"])

        print(f"\n🎮 Dificultad: {difficulty} | max_errors={cfg['max_errors']} | noise={cfg['noise']}")
        print("El agente comenzará a jugar. Observa la ventana del ahorcado.\n")

        play_visual_graphic_spacious(
            target,
            agent,
            max_errors=cfg["max_errors"],
            sleep=1.0,
            fig=fig,
            ax=ax_main
        )

    # ----- Crear botones de dificultad -----
    ax_facil = fig.add_axes([0.05, 0.05, 0.18, 0.08])
    ax_medio = fig.add_axes([0.28, 0.05, 0.18, 0.08])
    ax_dificil = fig.add_axes([0.51, 0.05, 0.18, 0.08])
    ax_imposible = fig.add_axes([0.74, 0.05, 0.18, 0.08])

    btn_facil = Button(ax_facil, "Fácil")
    btn_medio = Button(ax_medio, "Medio")
    btn_dificil = Button(ax_dificil, "Difícil")
    btn_imposible = Button(ax_imposible, "Imposible")

    btn_facil.on_clicked(lambda event: start_game("facil"))
    btn_medio.on_clicked(lambda event: start_game("medio"))
    btn_dificil.on_clicked(lambda event: start_game("dificil"))
    btn_imposible.on_clicked(lambda event: start_game("imposible"))

    # Mostrar ventana (el programa vive mientras la ventana esté abierta)
    plt.show(block=True)


if __name__ == "__main__":
    main()
