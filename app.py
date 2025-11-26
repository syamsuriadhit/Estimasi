import os, re
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

df = None
years = []
history = []

# =============================
# HELPER
# =============================
def safe_name(name):
    return re.sub(r'[^0-9a-zA-Z_-]', '_', name)

def predict_series(series, method="linear", future=5):
    X = np.arange(len(series)).reshape(-1, 1)
    y = np.array(series, dtype=float)

    if method == "linear":
        model = LinearRegression()
        model.fit(X, y)
        Xf = np.arange(len(series), len(series) + future).reshape(-1, 1)
        pred = model.predict(Xf)

    elif method == "polynomial":
        poly = PolynomialFeatures(degree=3)
        Xp = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(Xp, y)
        Xf = np.arange(len(series), len(series) + future).reshape(-1, 1)
        pred = model.predict(poly.transform(Xf))

    elif method == "random_forest":
        model = RandomForestRegressor(n_estimators=150, random_state=0)
        model.fit(X, y)
        Xf = np.arange(len(series), len(series) + future).reshape(-1, 1)
        pred = model.predict(Xf)

    return pred


# =============================
# ROUTES
# =============================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global df, years
    file = request.files["file"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    if file.filename.endswith(("xlsx", "xls")):
        df = pd.read_excel(path)
    elif file.filename.endswith("csv"):
        df = pd.read_csv(path)
    elif file.filename.endswith("json"):
        df = pd.read_json(path)

    years = [str(c) for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
    return redirect("/preview")


@app.route("/preview")
def preview():
    global df
    if df is None:
        return redirect("/")

    preview_table = df.head(10).to_html(classes="table table-bordered", index=False)
    return render_template("preview.html", df=df, preview=preview_table)


@app.route("/estimasi", methods=["POST"])
def estimasi():
    global df, years, history

    country = request.form["country"]
    method = request.form["method"]
    future = int(request.form["future"])

    row = df[df["Country Name"] == country]
    if row.empty:
        return "Negara tidak ditemukan"

    series = row[years].values.flatten()
    series = pd.to_numeric(series, errors="coerce")
    series = pd.Series(series).fillna(method="ffill").fillna(method="bfill")

    pred = predict_series(series.tolist(), method, future)

    tahun_terakhir = int(years[-1])
    future_years = [tahun_terakhir + i for i in range(1, future + 1)]

    hasil = list(zip(future_years, pred))

    # ========================
    # BUAT GRAFIK
    # ========================
    plt.figure(figsize=(8, 4))
    plt.plot([int(y) for y in years], series, label="Data Asli")
    plt.plot(future_years, pred, '--', label="Estimasi")
    plt.legend()
    plt.title(country + " - Estimasi")
    plt.xlabel("Tahun")
    plt.ylabel("Nilai")

    graph_name = safe_name(country) + ".png"
    graph_path = os.path.join(STATIC_FOLDER, graph_name)
    plt.savefig(graph_path)
    plt.close()

    history.append({
        "country": country,
        "method": method,
        "hasil": hasil,
        "graph": graph_name
    })

    return render_template("hasil.html",
                           country=country,
                           method=method,
                           hasil=hasil,
                           graph=graph_name)


@app.route("/history")
def show_history():
    return render_template("history.html", history=history)


@app.route("/download")
def download():
    if not history:
        return "Belum ada riwayat"

    rows = []
    for h in history:
        for th, val in h["hasil"]:
            rows.append({
                "Negara": h["country"],
                "Metode": h["method"],
                "Tahun": th,
                "Estimasi": val
            })

    df_h = pd.DataFrame(rows)
    path = "static/history.csv"
    df_h.to_csv(path, index=False)
    return redirect("/" + path)


# =============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

