from flask import (
    Flask,
    jsonify,
    render_template,
    request
)

from api.predict import predict_uploaded_file, is_allowed_file


app = Flask(
    __name__
)

app.config[
    "MAX_CONTENT_LENGTH"
] = 5 * 1024 * 1024


@app.route(
    "/",
    methods=["GET"]
)
def home():
    """
    Display the image classification interface.
    """

    return render_template(
        "index.html"
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    if not is_allowed_file(file.filename):
        return jsonify(
            {"error": "Unsupported file type. Use PNG, JPG or JPEG."}
        ), 400

    try:
        predictions = predict_uploaded_file(file)

        return jsonify(
            {
                "predictions": predictions
            }
        )

    except Exception as error:
        return jsonify(
            {
                "error": str(error)
            }
        ), 500


@app.errorhandler(
    413
)
def request_entity_too_large(error):

    return jsonify({
        "error": "Image file is too large. Maximum size is 5 MB."
    }), 413


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )