import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user,
)
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb

# Flask setup
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = "./uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Database setup
db = SQLAlchemy(app)

# Login manager setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ChromaDB setup
client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection("chatbot_docs")

# HuggingFace model setup
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")


# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)


# Chatbot model
class Chatbot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    creation_date = db.Column(db.DateTime, default=db.func.now())


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Routes
@app.route("/")
def home():
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        if User.query.filter(
            (User.username == username) | (User.email == email)
        ).first():
            flash("Username or Email already exists!")
        else:
            user = User(username=username, email=email, password=password)
            db.session.add(user)
            db.session.commit()
            flash("Registration successful. Please log in.")
            return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email_or_username = request.form["email_or_username"]
        password = request.form["password"]
        user = User.query.filter(
            (User.email == email_or_username) | (User.username == email_or_username)
        ).first()
        if user and user.password == password:
            login_user(user)
            session.permanent = True
            app.permanent_session_lifetime = 86400  # 24 hours
            return redirect(url_for("dashboard"))
        flash("Invalid credentials.")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    chatbots = Chatbot.query.filter_by(user_id=current_user.id).all()
    return render_template("dashboard.html", chatbots=chatbots)


@app.route("/wizard", methods=["GET", "POST"])
@login_required
def wizard():
    if request.method == "POST":
        step = int(request.form.get("step", 1))
        if step == 1:
            # Save chatbot basic info
            name = request.form["name"]
            description = request.form["description"]
            chatbot = Chatbot(
                user_id=current_user.id, name=name, description=description
            )
            db.session.add(chatbot)
            db.session.commit()
            session["chatbot_id"] = chatbot.id
            flash("Chatbot basic information saved!")
        elif step == 2:
            # Save personality setup (placeholder for this step)
            flash("Personality setup saved!")
        elif step == 3:
            # Upload and process document
            file = request.files["file"]
            if file and file.filename.split(".")[-1].lower() in [
                "pdf",
                "txt",
                "doc",
                "docx",
            ]:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)
                process_document(filepath, chatbot_id=session["chatbot_id"])
                flash("Document uploaded and processed successfully!")
            else:
                flash(
                    "Invalid file type. Only PDF, TXT, DOC, and DOCX files are allowed."
                )
        return redirect(url_for("wizard", step=step + 1))
    return render_template("wizard.html")


def process_document(filepath, chatbot_id):
    """Processes the uploaded document and stores embeddings in ChromaDB."""
    extension = filepath.split(".")[-1].lower()
    if extension == "pdf":
        loader = PyPDFLoader(filepath)
    elif extension in ["doc", "docx"]:
        loader = UnstructuredWordDocumentLoader(filepath)
    elif extension == "txt":
        loader = TextLoader(filepath)
    else:
        raise ValueError("Unsupported file type")

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    for chunk in chunks:
        collection.add(
            ids=[str(uuid.uuid4())],
            metadatas=[{"chatbot_id": chatbot_id}],
            documents=[chunk.page_content],
            embeddings=[embeddings.embed(chunk.page_content)],
        )


if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Ensure the database tables are created within the app context
    app.run(debug=True)
