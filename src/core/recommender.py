class TrendResearcher:
    """
    Symulator modułu badawczego. 
    W wersji produkcyjnej ta klasa łączyłaby się z Google Trends API lub Pinterest API.
    Obecnie korzysta z predefiniowanej bazy trendów na rok 2025/2026.
    """
    def __init__(self):
        self.trending_db = {
            "Oval": ["Butterfly Cut", "Skinny Lob", "Italian Bob"],
            "Square": ["Wolf Cut", "Wispy Bangs Shag", "Textured Pixie"],
            "Round": ["Curtain Bangs", "Long Layered Hush Cut", "Deep Side Part"],
            "Heart": ["Bixie (Bob+Pixie)", "Chin-length Bob", "Piecey Bangs"],
            "Oblong": ["Bottleneck Bangs", "Short Curly Shag", "French Bob"]
        }

    def fetch_trends(self, face_shape):
        """Pobiera najnowsze trendy dla danego kształtu."""
        return self.trending_db.get(face_shape, [])

class HairstyleRecommender:
    def __init__(self):
        # Baza wiedzy rozdzielona na płeć
        self.database = {
            "Oval": {
                "description": "Twarz owalna jest idealnie zbalansowana. To 'uniwersalny' kształt.",
                "Female": [
                    "Long Waves (Długie fale)",
                    "Blunt Bob (Bob z prostym cięciem)",
                    "Pixie Cut (Krótkie cięcie)",
                    "High Ponytail (Wysoki kucyk)"
                ],
                "Male": [
                    "Pompadour (Wysoko zaczesane)",
                    "Buzz Cut (Krótkie wojskowe)",
                    "Quiff (Zaczes do tyłu)",
                    "Faux Hawk (Irokez stylizowany)"
                ],
                "avoid": "Unikaj grzywek, które całkowicie zasłaniają czoło i zaburzają idealne proporcje."
            },
            "Square": {
                "description": "Mocna linia szczęki i szerokie czoło. Celem jest złagodzenie rysów.",
                "Female": [
                    "Long Bob (Lob)",
                    "Side-swept Bangs (Grzywka na bok)",
                    "Soft Layers (Miękkie cieniowanie)",
                    "Wavy Hair (Fale)"
                ],
                "Male": [
                    "Crew Cut (Krótko po bokach)",
                    "Undercut (Wycięte boki)",
                    "Side Part (Przedziałek na bok)",
                    "Slicked Back (Zaczesane gładko)"
                ],
                "avoid": "Unikaj geometrycznych, ostrych cięć kończących się równo z linią szczęki."
            },
            "Round": {
                "description": "Twarz ma podobną szerokość i wysokość. Celem jest jej optyczne wydłużenie.",
                "Female": [
                    "Long Straight Hair (Długie proste)",
                    "Deep Side Part (Głęboki przedziałek)",
                    "High Bun (Wysoki kok)",
                    "Textured Lob"
                ],
                "Male": [
                    "Pompadour (Wydłuża twarz)",
                    "Faux Hawk (Dodaje wysokości)",
                    "Side Part z objętością",
                    "Angular Fringe (Kanciasta grzywka)"
                ],
                "avoid": "Unikaj gładkich fryzur przylegających do głowy i cięć kończących się przy policzkach."
            },
            "Heart": {
                "description": "Szerokie czoło i wąska broda. Celem jest balans dołu twarzy.",
                "Female": [
                    "Chin-length Bob (Bob do brody)",
                    "Side Swept Bangs",
                    "Waves (Fale od połowy)",
                    "Low Ponytail"
                ],
                "Male": [
                    "Medium Length Sweep",
                    "Side Part (Dłuższy)",
                    "Long Fringe (Długa grzywka)",
                    "Layered Cut"
                ],
                "avoid": "Unikaj dużej objętości na samym czubku głowy (skracanie twarzy)."
            },
            "Oblong": {
                "description": "Twarz pociągła. Celem jest optyczne skrócenie i poszerzenie.",
                "Female": [
                    "Curtain Bangs (Grzywka kurtynowa)",
                    "Curly/Wavy (Loki dodają szerokości)",
                    "Short Bob",
                    "Side Part"
                ],
                "Male": [
                    "Buzz Cut",
                    "Side Part (Klasyczny)",
                    "Caesar Cut (Krótka grzywka)",
                    "Short Crop"
                ],
                "avoid": "Unikaj bardzo wysokich upięć i długich prostych włosów bez objętości."
            }
        }

    def get_advice(self, face_shape, gender="Female"):
        """
        Zwraca poradę dla danego kształtu i płci.
        gender: 'Female' lub 'Male'
        """
        data = self.database.get(face_shape, {
            "description": "Nierozpoznany kształt.",
            "Female": [],
            "Male": [],
            "avoid": "-"
        })

        return {
            "description": data["description"],
            "hairstyles": data.get(gender, []),
            "avoid": data["avoid"]
        }