from flask import render_template

from app import app, recommender


@app.route("/")
def home():
    style = app.url_for("static", filename="style.css")
    out_str = ""
    try:
        for value in recommender.low["Recommendations"].values[0]:
            out_str = out_str + "\r\n" + value
    except Exception as e:
        print("something f'd up", e)
        out_str = "Hello, World!"
    return render_template("index.html", content=out_str, style=style)
