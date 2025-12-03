# ZRSVN RAG Preprocessing  
  
Sistem za predprocesiranje dokumentov za RAG (Retrieval Augmented Generation) aplikacije.  
  
## O projektu  
  
Sistem implementira štiri-fazni pipeline za predprocesiranje PDF dokumentov:  
  
- **Faza 1**: Razčlenjevanje PDF dokumentov in generiranje JSON struktur.  
- **Faza 2**: Predobdelava podatkov in vstavljanje v PostgreSQL bazo.
- **Faza 3**: Generiranje metapodatkov (ključne besede, povzetki, opisi) z LLMi.
- **Faza 4**: Generiranje vektorskih vložitev za semantično iskanje.
  
## Funkcionalnosti  
  
- Prenos PDF dokumentov iz S3/MinIO shrambe.
- Ekstrakcija besedil, slik in tabel iz PDFjev z Docling.  
- Segmentacija besedila na optimalno dolžino.  
- LLM generirane ključne besede in povzetki.  
- Zaznavanje jezika besedilnih blokov.  
- Generiranje 768-dimenzionalnih vložitev (privzeto z BAAI/bge-m3 modelom).  
- Hierarhična struktura metapodatkov (dokument → sekcija → element).  
  
## Tehnične zahteve  
  
- Python 3.8+.  
- PostgreSQL baza s shemo `rag_najdbe`.
- MinIO/S3 shramba.  
- Idealno CUDA-kompatibilna grafična kartica (za hitrejše ustvarjanje vložitev).  
- Azure OpenAI API dostop.  
  
## Namestitev  
  
1. Klonirajte repozitorij:  
```bash  
git clone https://github.com/gregorgatej/zrsvn-rag-preprocessing.git  
cd zrsvn-rag-preprocessing
```
2. Namestite odvisnosti:
```bash  
pip install -r requirements.txt
```
3. Ustvarite .env datoteko z naslednjimi spremenljivkami:
```bash  
S3_ACCESS_KEY=tvoj_s3_access_key  
S3_SECRET_ACCESS_KEY=tvoj_s3_secret_key  
POSTGRES_PASSWORD=tvoj_postgres_password  
AZURE_OPENAI_API_KEY=tvoj_azure_openai_key  
AZURE_OPENAI_ENDPOINT=tvoj_azure_endpoint
```
4. Pripravite PostgreSQL bazo s shemo rag_najdbe.

## Uporaba

### Zagon celotnega pipeline-a

python pipeline/all_phases_flow.py

### Zagon posameznih faz

python pipeline/phase1_flow.py  
  
python pipeline/phase2_flow.py  
  
python pipeline/phase3_flow.py  
  
python pipeline/phase4_flow.py

## Struktura podatkov

Sistem ustvari hierarhično strukturo podatkov v PostgreSQL bazi :

- files - osnovni metapodatki dokumentov.
- sections - sekcije znotraj dokumentov.
- section_elements - posamezni elementi (odstavki, slike, tabele).
- text_chunks - optimizirani besedilni bloki za RAG.
- embeddings - vektorske reprezentacije za semantično iskanje.

## Orkestracija

Pipeline uporablja Prefect za upravljanje podatkovnih tokov z vgrajeno podporo za:

- Avtomatsko ponovitev ob napakah (3x z 2s zamikom).
- Sledenje napredka z JSON datotekami.
- Zaporedno izvajanje faz.