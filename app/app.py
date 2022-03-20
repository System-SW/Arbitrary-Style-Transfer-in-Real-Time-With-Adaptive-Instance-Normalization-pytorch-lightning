import streamlit as st
from inferance import Model
from utils import TITLE, DESC, AUTHOR, URL, MODEL_PATH, EXT, image_load

st.header(TITLE)
st.text(DESC)

try:
    model_load_state = st.text("Loading Model...")
    model = Model(MODEL_PATH)
    model_load_state.text("Loading Model...done!")
except Exception as e:
    model_load_state.text("loading is failed! Check your model!" + str(e))


content, style = st.columns(2)

with content:
    st.write("## Content Image...")
    content_image_file = st.file_uploader("Pick a Content image", type=EXT)
    try:
        content_image = image_load(content_image_file)
        st.image(content_image, caption="Content Image")
    except Exception:
        pass


with style:
    st.write("## Style Image...")
    style_image_file = st.file_uploader("Pick a Style image", type=EXT)
    try:
        style_image = image_load(style_image_file)
        st.image(style_image, caption="Style Image")
    except Exception:
        pass


with st.form("Control panel"):
    alpha = st.slider("Alpha", min_value=0.0, max_value=1.0, value=0.5)

    content_size = st.select_slider(
        "content size",
        [128, 256, 512, 1024],
        value=1024,
    )
    style_size = st.select_slider(
        "style size",
        [128, 256, 512, 1024, "No Resize"],
        value=128,
    )
    predict = st.form_submit_button("RUN")

    if predict:
        image = model.predict(
            content=content_image,
            style=style_image,
            alpha=alpha,
            content_size=content_size,
            style_size=style_size,
        )
        st.session_state["image"] = image


if "image" in st.session_state:
    st.write("Resultant Image...")
    st.image(st.session_state["image"])


st.write(f"Made by {AUTHOR}")
st.markdown(f""" Source code : [link]({URL}) """)
