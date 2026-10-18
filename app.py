import os
import re
import uuid

from flask import (
    Flask,
    render_template,
    redirect,
    url_for,
    request,
    flash,
    send_from_directory,
    abort,
    jsonify,
)
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user,
)
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

from utils.extensions import db
from utils.models import User, Prediction

load_dotenv()

app = Flask(__name__, instance_relative_config=True)
os.makedirs(app.instance_path, exist_ok=True)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
if not app.config["SECRET_KEY"]:
    raise RuntimeError("SECRET_KEY is missing. Add it to your .env file.")

app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///" + os.path.join(app.instance_path, "app.db")
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

UPLOAD_FOLDER = os.path.join(app.instance_path, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access your dashboard."
login_manager.login_message_category = "error"


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


with app.app_context():
    db.create_all()


# -----------------------------
# Validation helpers
# -----------------------------
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_registration(name, email, password, confirm_password):
    errors = []
    if not name or not name.strip():
        errors.append("Please enter your name.")
    if not email or not EMAIL_REGEX.match(email.strip()):
        errors.append("Please enter a valid email address.")
    if not password or len(password) < 8:
        errors.append("Password must be at least 8 characters long.")
    if password != confirm_password:
        errors.append("Passwords do not match.")
    return errors


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# -----------------------------
# Auth routes
# -----------------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    errors = []

    if request.method == "POST":
        name = request.form.get("name", "")
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        errors = validate_registration(
            name, email, password, confirm_password
        )

        if not errors:
            existing = User.query.filter_by(email=email).first()
            if existing:
                errors.append(
                    "An account with this email already exists."
                )

        if not errors:
            user = User(name=name.strip(), email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash(
                "Account created successfully. Please log in.",
                "success",
            )
            return redirect(url_for("login"))

        for error in errors:
            flash(error, "error")

    return render_template("register.html", errors=errors)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not EMAIL_REGEX.match(email):
            flash("Please enter a valid email address.", "error")
        elif not password:
            flash("Please enter your password.", "error")
        else:
            user = User.query.filter_by(email=email).first()
            if user and user.check_password(password):
                login_user(user)
                flash(
                    f"Welcome back, {user.name}! You are now logged in.",
                    "success",
                )
                next_page = request.args.get("next")
                if next_page and next_page.startswith("/"):
                    return redirect(next_page)
                return redirect(url_for("dashboard"))
            else:
                flash("Invalid email or password.", "error")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


# -----------------------------
# Dashboard / profile
# -----------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    from utils.disease_info import DISPLAY_NAMES

    predictions = (
        Prediction.query.filter_by(user_id=current_user.id)
        .order_by(Prediction.created_at.desc())
        .limit(6)
        .all()
    )
    return render_template(
        "dashboard.html",
        predictions=predictions,
        DISPLAY_NAMES=DISPLAY_NAMES,
    )


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    errors = []

    if request.method == "POST":
        action = request.form.get("action", "")

        if action == "update_name":
            new_name = request.form.get("name", "").strip()

            if not new_name:
                errors.append("Name cannot be empty.")
            elif new_name == current_user.name:
                errors.append("That is already your current name.")
            else:
                current_user.name = new_name
                db.session.commit()
                flash("Your name has been updated.", "success")
                return redirect(url_for("profile"))

        elif action == "update_password":
            current_password = request.form.get("current_password", "")
            new_password = request.form.get("new_password", "")
            confirm_password = request.form.get("confirm_password", "")

            if not current_user.check_password(current_password):
                errors.append("Your current password is incorrect.")
            elif len(new_password) < 8:
                errors.append(
                    "New password must be at least 8 characters long."
                )
            elif new_password != confirm_password:
                errors.append("New passwords do not match.")
            else:
                current_user.set_password(new_password)
                db.session.commit()
                flash("Your password has been updated.", "success")
                return redirect(url_for("profile"))

    for error in errors:
        flash(error, "error")

    return render_template("profile.html", errors=errors)


# -----------------------------
# Classification
# -----------------------------
@app.route("/predict", methods=["GET"])
@login_required
def predict():
    return render_template("predict.html")


@app.route("/api/classify", methods=["POST"])
@login_required
def api_classify():
    """
    Accepts an uploaded image, runs inference, saves a Prediction
    record, and returns JSON for the inline results UI.
    """
    file = request.files.get("image")

    if not file or file.filename == "":
        return jsonify({"error": "Please choose an image file."}), 400

    if not allowed_file(file.filename):
        return (
            jsonify({
                "error": "Unsupported file type. Use PNG, JPG, JPEG, or WEBP."
            }),
            400,
        )

    original_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    save_path = os.path.join(
        app.config["UPLOAD_FOLDER"], unique_name
    )
    file.save(save_path)

    try:
        from utils.classifier import predict_image
        result = predict_image(save_path)
    except Exception as exc:
        app.logger.exception("Classification failed")
        return (
            jsonify({"error": f"Classification failed: {exc}"}),
            500,
        )

    prediction = Prediction(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        image_filename=unique_name,
        user_id=current_user.id,
    )
    db.session.add(prediction)
    db.session.commit()

    from utils.disease_info import (
        DISPLAY_NAMES,
        get_disease_info,
    )

    predicted_key = result["predicted_class"]
    predicted_label = DISPLAY_NAMES.get(predicted_key, predicted_key)

    # Build ordered score list: predicted class first, then the
    # remaining classes in the model's normal (CLASS_NAMES) order.
    from utils.disease_info import CLASS_NAMES

    scores = result["scores"]
    ordered = []
    ordered.append((
        predicted_key,
        DISPLAY_NAMES.get(predicted_key, predicted_key),
        float(scores.get(predicted_key, result["confidence"])),
    ))
    for name in CLASS_NAMES:
        if name == predicted_key:
            continue
        ordered.append((
            name,
            DISPLAY_NAMES.get(name, name),
            float(scores.get(name, 0.0)),
        ))

    info = get_disease_info(predicted_key)

    return jsonify({
        "prediction_id": prediction.id,
        "predicted_class": predicted_key,
        "predicted_label": predicted_label,
        "confidence": result["confidence"],
        "scores": [
            {
                "key": k,
                "label": label,
                "score": score,
            }
            for k, label, score in ordered
        ],
        "info": {
            "title": info["title"],
            "description": info["description"],
            "cause": info["cause"],
            "symptoms": info["symptoms"],
            "recommendation": info["recommendation"],
        },
        "image_url": url_for(
            "uploaded_file", filename=unique_name
        ),
        "created_at": prediction.created_at.strftime(
            "%b %d, %Y · %H:%M"
        ),
    })


@app.route("/result/<int:prediction_id>")
@login_required
def result(prediction_id):
    """Standalone result page (used from dashboard/history links)."""
    prediction = db.session.get(Prediction, prediction_id)
    if not prediction or prediction.user_id != current_user.id:
        abort(404)

    from utils.disease_info import (
        CLASS_NAMES,
        DISPLAY_NAMES,
        get_disease_info,
    )

    info = get_disease_info(prediction.predicted_class)

    # Try to recompute full scores from stored image.
    ordered_scores = []
    try:
        from utils.classifier import predict_image
        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"], prediction.image_filename
        )
        scores = predict_image(image_path)["scores"]
        predicted_key = prediction.predicted_class
        ordered_scores.append((
            predicted_key,
            DISPLAY_NAMES.get(predicted_key, predicted_key),
            float(scores.get(predicted_key, prediction.confidence)),
        ))
        for name in CLASS_NAMES:
            if name == predicted_key:
                continue
            ordered_scores.append((
                name,
                DISPLAY_NAMES.get(name, name),
                float(scores.get(name, 0.0)),
            ))
    except Exception:
        ordered_scores = [(
            prediction.predicted_class,
            DISPLAY_NAMES.get(
                prediction.predicted_class, prediction.predicted_class
            ),
            prediction.confidence,
        )]

    return render_template(
        "result.html",
        prediction=prediction,
        info=info,
        predicted_label=DISPLAY_NAMES.get(
            prediction.predicted_class, prediction.predicted_class
        ),
        ordered_scores=ordered_scores,
    )


@app.route("/history")
@login_required
def history():
    from utils.disease_info import DISPLAY_NAMES

    predictions = (
        Prediction.query.filter_by(user_id=current_user.id)
        .order_by(Prediction.created_at.desc())
        .all()
    )
    return render_template(
        "history.html",
        predictions=predictions,
        DISPLAY_NAMES=DISPLAY_NAMES,
    )


@app.route("/uploads/<path:filename>")
@login_required
def uploaded_file(filename):
    safe_name = secure_filename(filename)
    prediction = Prediction.query.filter_by(
        image_filename=safe_name,
        user_id=current_user.id,
    ).first()
    if not prediction:
        abort(404)
    return send_from_directory(
        app.config["UPLOAD_FOLDER"], safe_name
    )


@app.errorhandler(413)
def too_large(_error):
    if request.path.startswith("/api/"):
        return (
            jsonify({"error": "Image too large. Maximum size is 8 MB."}),
            413,
        )
    flash("That image is too large. Maximum size is 8 MB.", "error")
    return redirect(url_for("predict")), 303


if __name__ == "__main__":
    app.run(debug=True)