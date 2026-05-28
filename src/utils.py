from pathlib import Path


def save_uploaded_files(uploaded_files, folder: str = "data/uploaded_docs") -> str:
    """
    Saves Streamlit uploaded files to a local folder.
    """
    upload_folder = Path(folder)
    upload_folder.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:
        file_path = upload_folder / uploaded_file.name

        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

    return str(upload_folder)