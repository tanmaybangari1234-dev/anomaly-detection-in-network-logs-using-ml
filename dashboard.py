import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import joblib
from PIL import Image, ImageTk
import sqlite3
import os
import threading
# =========================
# DATABASE SETUP
# =========================
DB_NAME = "ids_logs.db"
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                level TEXT,
                message TEXT)""")
def insert_log(level, message):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            INSERT INTO logs (timestamp, level, message)
            VALUES (datetime('now'), ?, ?)
        """, (level, message))
def fetch_logs(limit=10):
    with sqlite3.connect(DB_NAME) as conn:
        return conn.execute("""
            SELECT timestamp, level, message
            FROM logs
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
init_db()
# =========================
# LOAD MODEL (SAME AS GUI)
# =========================
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("features.pkl")
label_encoders = joblib.load("label_encoders.pkl")
# =========================
# MAIN APP
# =========================
class IDSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intrusion Detection Dashboard")
        self.root.geometry("500x600")
        self.image_refs = []
        self.setup_scroll()
        self.build_ui()
        self.load_logs()
    # =====================
    # SCROLLABLE DASHBOARD
    # =====================
    def setup_scroll(self):
        container = tk.Frame(self.root)
        container.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(container)
        v_scroll = tk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        h_scroll = tk.Scrollbar(container, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set)
        v_scroll.pack(side="right", fill="y")
        h_scroll.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.main_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.main_frame, anchor="nw")
        self.main_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        # Mouse scroll
        self.canvas.bind_all("<MouseWheel>",
            lambda e: self.canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        self.canvas.bind_all("<Shift-MouseWheel>",
            lambda e: self.canvas.xview_scroll(int(-1*(e.delta/120)), "units"))
    # =====================
    # UI DESIGN
    # =====================
    def build_ui(self):
        tk.Label(self.main_frame,
                 text="Intrusion Detection System Dashboard",
                 font=("Arial", 18, "bold")).pack(pady=10)
        self.result_var = tk.StringVar()
        tk.Label(self.main_frame,
                 textvariable=self.result_var,
                 font=("Arial", 12)).pack()
        content = tk.Frame(self.main_frame)
        content.pack(anchor="nw")
        # LEFT SIDE → GRAPHS
        self.image_frame = tk.Frame(content)
        self.image_frame.grid(row=0, column=0, padx=10, sticky="nw")
        # RIGHT SIDE → ALERTS
        side = tk.Frame(content, width=350)
        side.grid(row=0, column=1, padx=20, sticky="n")
        side.grid_propagate(False)
        tk.Label(side, text="Recent Alerts",
                 font=("Arial", 12, "bold")).pack()
        self.log_box = tk.Text(side, width=45, height=25)
        self.log_box.pack()
        tk.Label(side, text="\nModel Info",
                 font=("Arial", 12, "bold")).pack()
        tk.Label(side,
                 text="Model: Random Forest\nDataset: UNSW-NB15\nType: Binary").pack()
        tk.Button(self.main_frame,
                  text="Upload Test File",
                  command=self.start_prediction,
                  width=25, height=2).pack(pady=20)
    # =====================
    # THREADING
    # =====================
    def start_prediction(self):
        threading.Thread(target=self.predict, daemon=True).start()
    # =====================
    # SHOW IMAGES
    # =====================
    def show_images(self):
        for w in self.image_frame.winfo_children():
            w.destroy()
        self.image_refs.clear()
        files = [
            "attack_distribution.png",
            "attack_vs_normal_packets.png",
            "class_distribution.png",
            "confusion_matrix.png",
            "duration_distribution.png",
            "optimized_correlation_heatmap.png",
            "roc_curve.png",
            "top_features_heatmap.png"]
        r = c = 0
        for path in files:
            frame = tk.Frame(self.image_frame, bd=1, relief="solid")
            frame.grid(row=r, column=c, padx=15, pady=15)
            if os.path.exists(path):
                img = Image.open(path).resize((600, 500))
                                               
                tk_img = ImageTk.PhotoImage(img)
                lbl = tk.Label(frame, image=tk_img)
                lbl.image = tk_img
                lbl.pack()
                self.image_refs.append(tk_img)
            else:
                tk.Label(frame, text=f"Missing:\n{path}").pack()
            c += 1
            if c == 2:
                c = 0
                r += 1
    # =====================
    # LOAD LOGS
    # =====================
    def load_logs(self):
        self.log_box.delete("1.0", tk.END)
        logs = fetch_logs()
        if not logs:
            self.log_box.insert(tk.END, "No logs available\n")
        else:
            for t, level, msg in logs:
                self.log_box.insert(tk.END, f"{t} | {level} | {msg}\n")
    # =====================
    # PREDICTION 
    # =====================
    def predict(self):
        file = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
        if not file:
            return
        try:
            df = pd.read_excel(file)
            # DROP UNUSED
            df.drop(columns=['id', 'attack_cat'], errors='ignore', inplace=True)
            # ENCODING 
            for col in label_encoders:
                if col in df.columns:
                    le = label_encoders[col]
                    df[col] = df[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
            # ALIGN FEATURES
            X = df.copy()
            for col in feature_columns:
                if col not in X.columns:
                    X[col] = 0
            X = X[feature_columns]
            # SCALE
            X_scaled = pd.DataFrame(
                scaler.transform(X),
                columns=feature_columns )
            # PREDICT
            preds = model.predict(X_scaled)
            normal = (preds == 0).sum()
            attack = (preds == 1).sum()
            total = len(preds)
            self.result_var.set(
                f"Normal: {normal} ({normal/total:.1%}) | "
                f"Attack: {attack} ({attack/total:.1%})" )
            # LOG
            insert_log("INFO", f"Attack={attack}, Normal={normal}")
            # UPDATE UI
            self.root.after(0, self.show_images)
            self.root.after(0, self.load_logs)
        except Exception as e:
            insert_log("ERROR", str(e))
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
# =====================
# RUN APP
# =====================
if __name__ == "__main__":
    root = tk.Tk()
    app = IDSApp(root)
    root.mainloop()