# Tamil History RAG — Knowledge Engine

A scholarly RAG (Retrieval-Augmented Generation) system over 5,000+ years of Tamil and Tamil Nadu history.

**🚀 Live demo:** [tamil-history-rag-jqwhjvvwppvjyl7zrnkhmi.streamlit.app](https://tamil-history-rag-jqwhjvvwppvjyl7zrnkhmi.streamlit.app/)

## Corpus

10 deep scholarly chapters covering:

| Chapter | Period | Topics |
|---------|--------|--------|
| 01 Prehistoric | 300,000 BP – 300 BCE | Attirampakkam, Keezhadi, Megalithic culture |
| 02 Sangam Age | 300 BCE – 300 CE | Sangam literature, Three Crowned Kings, Roman trade |
| 03 Pallava Dynasty | 275–897 CE | Temple architecture, Bhakti movement, Mahabalipuram |
| 04 Chola Empire | 848–1279 CE | Naval expeditions, Brihadeeswarar, village democracy |
| 05 Pandya Dynasty | 300 BCE – 1310 CE | Pearl fisheries, Madurai, Marco Polo |
| 06 Vijayanagara & Nayaks | 1336–1736 CE | Nayak kingdoms, Carnatic music, Bharatanatyam |
| 07 Colonial Period | 1498–1947 | Portuguese, Dutch, British, freedom fighters |
| 08 Dravidian Movement | 1925–present | Periyar, DMK, Anti-Hindi agitations |
| 09 Tamil Language | 300 BCE–present | Tolkappiyam, Sangam corpus, Thirukkural |
| 10 Modern Tamil Nadu | 1947–present | Politics, economy, cinema, diaspora |

## Stack

- **Retrieval:** Pure Python TF-IDF cosine similarity (zero heavy dependencies)
- **Generation:** Groq LLaMA-3.3-70b-versatile
- **Interface:** Streamlit
- **Chunking:** Section-level (H2 header boundaries) for optimal RAG quality

## The thesis

This project demonstrates how **documentation information architecture affects RAG quality**. Each corpus chunk is:
- Structured with clear Q&A headers for semantic retrieval
- Sized at the section level (not paragraph, not full document) for optimal context density
- Written with consistent terminology for reliable vector matching

Poor IA = poor retrieval = poor answers. This corpus is built to prove the opposite.

---

Built by [Azariah Onyx](https://github.com/AzariahOnyx) · [Portfolio](https://azariahonyx.github.io) · [LinkedIn](https://linkedin.com/in/onyx-aj)

