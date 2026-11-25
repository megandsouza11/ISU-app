"""
Step 2 - Streamlit app. Run: streamlit run streamlit_app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Lookalike - visual fashion search", layout="wide")

### load embeddings and model
@st.cache_data
def load_catalog():
  catalog = pd.read_csv("artifacts/catalog.csv")
  embeddings = np.load("artifacts/embedding.npy").astype("float32")
  return catalog, embeddings

@st.cache_resource
def load_model():
  return SentenceTransformer("clip-ViT-B-32")

catalog, embeddings = load_catalog()
model = load_model()

### recommender core
def search(query_vec, k, mask, exclude_id=None):
  """Cosine similarity = dot product, bc all vectors are normalized."""
  scores = embeddings @ query_vec
  scores[~mask] = -np.inf  # apply sidebar filters
  if exclude_id is not None:
    scores[catalog["id"].values == exclude_id] = -np.inf
  top = np.argsort(-scores)[:k]
  return catalog.iloc[top].assign(score=scores[top])

def show_grid(results, key_prefix, cols=5):
  columns = st.columns(cols)
  for i, (_, row) in enumerate(results.iterrows()):
    with columns[i % cols]:
      st.image(row["image_path"], use_container_width=True)
      st.caption(row["productDisplayName"])
      if "score" in row:
        st.caption(f"similarity {row['score']:.2f}")
      if st.button("More like this", key=f"{key_prefix}-{row['id']}"):
          st.session_state["selected_id"] = int(row["id"])
          st.rerun()

### sidebar filters
st.sidebar.header("Filters")
genders = st.sidebar.multiselect("Gender", sorted(catalog["gender"].dropna().unique()))
k = st.sidebar.slider("Results", 5, 30, 10, step=5)

mask = np.ones(len(catalog), dtype=bool)
if genders:
  mask &= catalog["gender"].isin(genders).values

if categories:
  mask &= catalog["masterCategory"].isin(categories).values


### more like this panel
st.title('👕 Lookalike')
st.write('Find fashion items that *look* similair - search by photo, by description, or by browsing.')

selected = st.session_state.get("selected_id")
if selected is not None:
  item = catalog[catalog["id"] == selected].iloc[0]
  idx = item.name
  st.subheader("Because you picked")
  left, right = st.columns([1,4])
  left.image(item["image_path"], use_container_width=True)
  right.markdown(f"**{item['productDisplayName']}**\n{item['articleType']} · {item['baseColour']}")
  if right.button("Clear"):
    del st.session_state["selected_id"]
    st.rerun()
  show_grid(search(embeddings[idx], k, mask, exclude_id=selected), "similair")
  st.divider()


### three inputs
tab_photo, tab_text, tab_browse = st.tabs(["📷 Upload a photo", "✍️ Describe it", "🛍️ Browse"])

with tab_photo:
  upload = st.file_uploader("Upload a photo of a clothing item", type=["jpg", "jpeg", "png"])
  if upload:
    image = Image.open(upload).convert("RGB")
    st.image(image, width=200)
    vec = model.encode(image, normalize_embeddings=True)
    show_grid(search(vec, k, mask), "photo")

with tab_text:
  query = st.text_input("e.g. 'red floral summer dress' or chunky white sneakers'")
  if query:
    vec = model.encode(query, normalize_embeddings=True)
    show_grid(search(vec, k, mask), "text")
  
with tab_browse:
  if st.button("🔀 Shuffle"):
    st.session_state["seed"] = np.random.randint(1_000_000)
  pool = catalog[mask]
  sample = pool.sample(min(k, len(pool)), random_state = st.session_state.get("seed", 0))
  show_grid(sample, "browse")
