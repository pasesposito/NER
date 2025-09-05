import streamlit as st
import spacy
from transformers import pipeline
from rdflib import Graph, Literal, RDF, RDFS, OWL, URIRef, Namespace
import networkx as nx
from pyvis.network import Network
import tempfile
import os

# -----------------------
# Caricamento modelli
# -----------------------
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_hf_model():
    return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# -----------------------
# Namespace Ontologia
# -----------------------
SHE = Namespace("http://example.org/she#")

# Mapping etichette NER -> Classi OWL
def map_entity_label(label: str) -> URIRef:
    mapping = {
        "PERSON": SHE.Person,
        "PER": SHE.Person,
        "ORG": SHE.Organization,
        "GPE": SHE.Location,
        "LOC": SHE.Location,
        "DATE": SHE.Date,
        "MISC": SHE.MiscEntity,
    }
    return mapping.get(label, SHE.Entity)

# -----------------------
# Ontologia di base
# -----------------------
def add_base_ontology(g: Graph):
    g.bind("she", SHE)
    g.add((SHE.Entity, RDF.type, OWL.Class))
    g.add((SHE.Person, RDF.type, OWL.Class)); g.add((SHE.Person, RDFS.subClassOf, SHE.Entity))
    g.add((SHE.Organization, RDF.type, OWL.Class)); g.add((SHE.Organization, RDFS.subClassOf, SHE.Entity))
    g.add((SHE.Location, RDF.type, OWL.Class)); g.add((SHE.Location, RDFS.subClassOf, SHE.Entity))
    g.add((SHE.Date, RDF.type, OWL.Class)); g.add((SHE.Date, RDFS.subClassOf, SHE.Entity))
    g.add((SHE.MiscEntity, RDF.type, OWL.Class)); g.add((SHE.MiscEntity, RDFS.subClassOf, SHE.Entity))
    g.add((SHE.Document, RDF.type, OWL.Class))
    g.add((SHE.hasEntity, RDF.type, OWL.ObjectProperty))
    g.add((SHE.hasText, RDF.type, OWL.DatatypeProperty))
    g.add((SHE.hasScore, RDF.type, OWL.DatatypeProperty))
    return g

# -----------------------
# Costruzione grafi RDF
# -----------------------
def build_graph_spacy(doc):
    g = Graph()
    add_base_ontology(g)
    doc_uri = URIRef(SHE["Document1"])
    g.add((doc_uri, RDF.type, SHE.Document))
    for ent in doc.ents:
        ent_uri = URIRef(SHE[ent.text.replace(" ", "_")])
        ent_class = map_entity_label(ent.label_)
        g.add((ent_uri, RDF.type, ent_class))
        g.add((doc_uri, SHE.hasEntity, ent_uri))
        g.add((ent_uri, SHE.hasText, Literal(ent.text)))
    return g

def build_graph_hf(entities):
    g = Graph()
    add_base_ontology(g)
    doc_uri = URIRef(SHE["Document1"])
    g.add((doc_uri, RDF.type, SHE.Document))
    for ent in entities:
        ent_uri = URIRef(SHE[ent['word'].replace(" ", "_")])
        ent_class = map_entity_label(ent['entity_group'])
        g.add((ent_uri, RDF.type, ent_class))
        g.add((doc_uri, SHE.hasEntity, ent_uri))
        g.add((ent_uri, SHE.hasText, Literal(ent['word'])))
        g.add((ent_uri, SHE.hasScore, Literal(ent['score'])))
    return g

# -----------------------
# Conversione RDF → NetworkX
# -----------------------
def rdf_to_networkx(g: Graph):
    G = nx.DiGraph()
    for s, p, o in g:
        G.add_node(str(s))
        G.add_node(str(o))
        G.add_edge(str(s), str(o), label=str(p).split("#")[-1])
    return G

def show_graph(g: Graph):
    nx_graph = rdf_to_networkx(g)
    net = Network(height="600px", width="100%", directed=True)

    # Colori per tipo entità
    color_map = {
        "Person": "blue",
        "Organization": "orange",
        "Location": "green",
        "Date": "yellow",
        "MiscEntity": "gray",
        "Entity": "gray",
        "Document": "red",
    }

    for node in nx_graph.nodes():
        label = node.split("#")[-1]
        # Determina colore
        color = "lightgray"
        for key, val in color_map.items():
            if key in label:
                color = val
                break
        net.add_node(node, label=label, color=color)

    for source, target, data in nx_graph.edges(data=True):
        net.add_edge(source, target, label=data["label"])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        return tmp.name

# -----------------------
# Interfaccia Streamlit
# -----------------------
st.title("🔎 NER → Ontology Graph (OWL/Turtle + Visualizzazione)")
st.write("Riconosci entità in un testo, genera un grafo RDF/OWL e visualizzalo con colori per tipo di entità.")

model_choice = st.radio("Seleziona il modello:", ("spaCy", "Hugging Face Transformers"))
text = st.text_area("Inserisci un testo in inglese:",
                    "Barack Obama was born in Hawaii in 1961. He became president in 2008.")

if st.button("Esegui NER e genera Grafo"):
    if model_choice == "spaCy":
        nlp = load_spacy_model()
        doc = nlp(text)
        st.subheader("📌 Risultati con spaCy")
        for ent in doc.ents:
            st.write(f"**{ent.text}** → {ent.label_}")
        graph = build_graph_spacy(doc)
    else:
        ner_pipeline = load_hf_model()
        entities = ner_pipeline(text)
        st.subheader("📌 Risultati con Hugging Face")
        for ent in entities:
            st.write(f"**{ent['word']}** → {ent['entity_group']} (score: {ent['score']:.2f})")
        graph = build_graph_hf(entities)

    # Serializza Turtle
    ttl_data = graph.serialize(format="turtle")
    st.subheader("📂 RDF/OWL Graph (Turtle)")
    st.code(ttl_data, language="turtle")
    st.download_button("⬇️ Scarica Turtle (.ttl)", data=ttl_data, file_name="ner_graph.ttl", mime="text/turtle")

    # Visualizzazione grafo
    st.subheader("🌐 Visualizzazione del Grafo (colori per tipo di entità)")
    graph_html = show_graph(graph)
    with open(graph_html, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=600, scrolling=True)
    os.remove(graph_html)
