from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, DirectoryLoader

## load all the text files from the directory
dir_loader=DirectoryLoader(
    "data/banking_rules",
    glob="**/*.pdf", ## Pattern to match files  
    loader_cls= PyMuPDFLoader, ##loader class to use
    show_progress=False

)

pdf_documents=dir_loader.load()
