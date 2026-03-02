# Koncepcja projektu

Czat AI oparty na architekturze wieloagentowej


## Back-end:
System wykorzystuje dwa mniejsze modele (worker) do generowania zróżnicowanych odpowiedzi oraz większy model (judge) do ich krytycznej oceny, weryfikacji i łączenia w ostateczny wynik. Całość będzie działała w klastrze AI stworzonym z dwóch PC. Dostęp do serwera backendowego będzie dostępny dla członków zespołu po VPN Tailscale z włączonym mechanizmem split DNS

# Klaster AI będzie sie składał z:
- PC1 - 4070ti 12GB VRAM + 64GB RAM 6400 - na tym systemie będą pracowały wstępnie dwa modele workerów.
- PC2 - 4060 8GB VRAM + 64GB RAM 5200 - na tym systemie będzie pracował model sędzi.
- Współpraca - LAN 2.5Gbps + API + VPN

Back-end jest w pełni skonteneryzowany przy użyciu Docker-compose. Komunikacja miedzy modelami będzie zarządzana poprzez API (FastAPI Python).

Silnik na którym będą pracowały modele to `llama.cpp (llama-cpp-python)`

Format modeli llm to `.gguf`

# Mechanizmy logiczne i tryby działania

System będzie posiadał trzy tryby:
- Fast - Użycie jednego modelu bez architektury multi-agent.
- Pro - Pełny cykl Worker-Judge oparty na modelach ogólnego przeznaczenia (General Use).
- Coding - Identyczny cykl jak "Pro", ale wykorzystujący wyspecjalizowane modele programistyczne (np. Coder) z naciskiem na jakość i bezpieczeństwo kodu.

Mechanizmy kognitywne - ślepa ocena ze zmianą kolejności: sędzia ocenia wygenerowane odpowiedzi bez wiedzy, która odpowiedz pochodzi od którego modelu. W przypadku wybrania pierwszej odpowiedzi zamiast drugiej z powodu, ze po prostu byla pierwsza, system zmienia kolejność i pyta ponownie.

Pętla refleksji - Sędzia nie tylko wybiera kod, ale może wygenerować konstruktywną krytykę i odesłać kod do Workera w celu poprawy (maksymalnie 2 iteracje, aby zapobiec nieskończonym pętlom – tzw. "Infinite Loops", na które skarżą się deweloperzy systemów wieloagentowych).

Sliding Window (Przesuwane Okno Kontekstowe) - Streszczanie lub ucinanie najstarszych wiadomości w długich konwersacjach, co chroni przed przekroczeniem limitów tokenów modelu i spadkiem wydajności.
