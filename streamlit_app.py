# CashQuest — Gamified Spending Coach (Hugging Face + Analytics + CSV upload in Sidebar)
# Run: streamlit run streamlit_app.py

import os, re
from datetime import date
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ============ CONFIG ============
load_dotenv()
st.set_page_config(page_title="CashQuest", page_icon="💰", layout="wide")

# Hugging Face auth
hf_token = os.getenv("HF_API_KEY")
if hf_token:
    try:
        from huggingface_hub import login
        login(hf_token)
    except Exception as e:
        st.sidebar.warning(f"Hugging Face login failed: {e}")

# ---------- THEME / CSS ----------
st.markdown("""
<style>
:root{ --bg:#000; --ink:#f2f2f2; }
.stApp{ background:var(--bg); color:var(--ink); }
.stApp::before{
  content:""; position:fixed; inset:0; z-index:-1;
  background-image:
    radial-gradient(circle at 8% 20%, rgba(255,215,0,.12) 2px, transparent 3px),
    radial-gradient(circle at 40% 80%, rgba(0,255,127,.10) 2px, transparent 3px),
    radial-gradient(circle at 78% 30%, rgba(135,206,235,.10) 2px, transparent 3px);
  background-size:120px 120px, 140px 140px, 160px 160px;
}
section[data-testid="stSidebar"] { background: #0a0a0a; }
.card{border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:16px; background:rgba(255,255,255,.03); }
.coupon{ border:1px dashed rgba(255,255,255,.3); padding:12px; border-radius:12px; margin: 6px 0;}
.coin { width:18px; height:18px; border-radius:50%;
        background: radial-gradient(circle at 30% 30%, #fff6b0, #ffd700 55%, #7a5f00);
        border:1px solid #ffe27a; display:inline-block; vertical-align:middle; margin-right:6px;}
.small{opacity:.85; font-size:.92rem;}
</style>
""", unsafe_allow_html=True)

# ============ SESSION STATE ============
def init_state():
    st.session_state.setdefault("coins", 0)
    st.session_state.setdefault("streak", 0)
    st.session_state.setdefault("last_logged", None)
    st.session_state.setdefault("daily_budget", 1500.0)
    st.session_state.setdefault("today_spend", 0.0)
    st.session_state.setdefault("coupons", [
        {"name":"10% off Groceries", "cost":100, "redeemed":False},
        {"name":"₹200 Ride Credit", "cost":250, "redeemed":False},
        {"name":"Movie Ticket", "cost":400, "redeemed":False},
    ])
    st.session_state.setdefault("history", [])
init_state()

CATEGORIES = ["Food", "Travel", "Shopping", "Bills", "Entertainment", "Health", "Education", "Other"]

# ============ UTIL: HISTORY DF ============
def history_df() -> pd.DataFrame:
    if not st.session_state.history:
        return pd.DataFrame(columns=["date","spend","budget","category"])
    df = pd.DataFrame(st.session_state.history)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["spend"] = pd.to_numeric(df["spend"], errors="coerce").fillna(0.0)
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce").fillna(0.0)
    df["category"] = df["category"].fillna("Other")
    return df

# ============ SPEND LOGIC ============
def log_entry(d: date, spend: float, budget: float, category: str):
    today = date.today()
    if d == today and st.session_state.get("last_logged") != today:
        st.session_state.today_spend = spend
        st.session_state.daily_budget = budget
        if spend <= budget:
            st.session_state.streak += 1
            st.session_state.coins += 10
        else:
            st.session_state.streak = 0
        st.session_state.last_logged = today

    replaced = False
    for row in st.session_state.history:
        if row["date"] == d:
            row["spend"], row["budget"], row["category"] = float(spend), float(budget), category
            replaced = True
            break
    if not replaced:
        st.session_state.history.append({
            "date": d, "spend": float(spend), "budget": float(budget), "category": category
        })

def redeem_coupon(i: int):
    c = st.session_state.coupons[i]
    if c["redeemed"]:
        st.info("Already redeemed"); return
    if st.session_state.coins < c["cost"]:
        st.error("Not enough coins yet."); return
    st.session_state.coins -= c["cost"]
    c["redeemed"] = True
    st.success(f"🎉 Enjoy your reward: {c['name']}!")

# ============ TEXT CLEANUP ============
def clean_answer(text: str) -> str:
    sentences = re.split(r'(?<=[.!?]) +', text)
    cleaned = []
    for s in sentences:
        s_strip = s.strip()
        if not s_strip: continue
        if cleaned and s_strip.lower().startswith(cleaned[-1][:18].lower()):
            continue
        if s_strip not in cleaned:
            cleaned.append(s_strip)
    if len(cleaned) < 2:
        cleaned.append("Start by setting aside a fixed amount each month in a savings account or recurring deposit.")
    if len(cleaned) < 3:
        cleaned.append("Track expenses by category to identify 1–2 areas to trim without changing your lifestyle.")
    return " ".join(cleaned[:5])

# ============ AI (Hugging Face only) ============
def ai_answer(question: str, persona: str = "student", model_name: str = None, max_new_tokens: int = 160) -> str:
    try:
        from transformers import pipeline
        model_name = model_name or os.getenv("HF_LOCAL_MODEL", "google/flan-t5-base")
        pipe = pipeline("text2text-generation", model=model_name)
        system = f"You are a concise, practical personal finance coach answering for a {persona}."
        prompt = f"{system}\nQuestion: {question}\nGive 4–6 short actionable points."
        raw = pipe(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"]
        return clean_answer(raw)
    except Exception as e:
        return f"❌ Hugging Face failed: {e}"

# ============ SIDEBAR ============
st.sidebar.title("💼 CashQuest")
menu = st.sidebar.radio(
    "📌 Navigate",
    ["Dashboard", "Log Daily Spend", "Monthly Analysis", "Yearly Trends", "Rewards", "AI Coach"],
    index=0
)

# Quick Stats
with st.sidebar.expander("Today / Quick Stats", expanded=True):
    st.write(f"**Coins:** {st.session_state.coins}")
    st.write(f"**Streak:** {st.session_state.streak} days")
    st.write(f"**Today's Spend:** ₹{st.session_state.today_spend:.0f}")
    st.write(f"**Daily Budget:** ₹{st.session_state.daily_budget:.0f}")

# CSV Tools (in Sidebar)
with st.sidebar.expander("📂 Data Tools", expanded=False):
    df = history_df()
    if not df.empty:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download History CSV", data=csv, file_name="cashquest_history.csv", mime="text/csv")
    # Upload
    sample_df = pd.DataFrame([{"date": date.today().isoformat(), "spend": 0, "budget": 0, "category": "Other"}])
    st.download_button("⬇️ Download Template CSV", data=sample_df.to_csv(index=False).encode("utf-8"),
                       file_name="cashquest_template.csv", mime="text/csv")
    upload_mode = st.radio("Import mode", ("Replace", "Append"), horizontal=True)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="history_upload_sidebar")
    if uploaded_file is not None:
        try:
            new_df = pd.read_csv(uploaded_file)
            needed = {"date","spend","budget","category"}
            if not needed.issubset(set(new_df.columns)):
                st.error("CSV must have columns: date, spend, budget, category")
            else:
                if upload_mode == "Replace":
                    st.session_state.history = []
                for _, r in new_df.iterrows():
                    st.session_state.history.append({
                        "date": pd.to_datetime(r["date"]).date(),
                        "spend": float(r["spend"]),
                        "budget": float(r["budget"]),
                        "category": str(r["category"]),
                    })
                st.success(f"✅ CSV imported ({upload_mode}). Entries: {len(new_df)}")
        except Exception as e:
            st.error(f"Failed to import CSV: {e}")

# Model options
with st.sidebar.expander("⚙️ AI Settings", expanded=False):
    hf_model_choice = st.selectbox(
        "Hugging Face model",
        ["google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-small"],
        index=0
    )
    max_tokens = st.slider("Max answer tokens", 80, 320, 160, step=10)

# ============ PAGES ============
st.title("💰 CashQuest")
st.caption("Build healthy money habits, track spending, earn rewards — and learn smarter ways to save.")

# --- Dashboard ---
if menu == "Dashboard":
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"<div class='card'><h3>Coins</h3><p class='small'>{st.session_state.coins}</p></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='card'><h3>Streak</h3><p class='small'>{st.session_state.streak} days</p></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='card'><h3>Today</h3><p class='small'>₹{st.session_state.today_spend:.0f}</p></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='card'><h3>Budget</h3><p class='small'>₹{st.session_state.daily_budget:.0f}</p></div>", unsafe_allow_html=True)

    df = history_df()
    st.markdown("### 📈 Recent Activity")
    if df.empty:
        st.info("No history yet. Log your first entry in **Log Daily Spend**.")
    else:
        df_sorted = df.sort_values("date").tail(14)
        st.line_chart(df_sorted.set_index("date")["spend"], height=220)
        st.bar_chart(df_sorted.set_index("date")[["spend","budget"]], height=240)

# --- Log Daily Spend ---
elif menu == "Log Daily Spend":
    st.subheader("🗓 Log a transaction")
    log_date = st.date_input("Date", value=date.today())
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        spend = st.number_input("Spend (₹)", min_value=0.0, value=float(st.session_state.today_spend), step=50.0)
    with col2:
        budget = st.number_input("Budget (₹)", min_value=0.0, value=float(st.session_state.daily_budget), step=50.0)
    with col3:
        category = st.selectbox("Category", CATEGORIES, index=0)
    if st.button("Save / Update"):
        log_entry(log_date, spend, budget, category)
        under = spend <= budget
        if log_date == date.today():
            st.progress(int((0 if budget==0 else min(spend/budget,1))*100), text="Today's budget usage")
        st.success("Entry saved. " + ("✅ Under budget! +10 coins" if (log_date==date.today() and under) else ""))

    st.markdown("### History")
    df = history_df().sort_values("date", ascending=False)
    st.dataframe(df, use_container_width=True)

# --- Monthly Analysis ---
elif menu == "Monthly Analysis":
    st.subheader("📅 Monthly Analysis")
    df = history_df()
    if df.empty:
        st.info("No data yet.")
    else:
        df["dt"] = pd.to_datetime(df["date"])
        df["year"] = df["dt"].dt.year
        df["month"] = df["dt"].dt.month
        sel_year = st.selectbox("Year", sorted(df["year"].unique(), reverse=True))
        months = sorted(df[df["year"]==sel_year]["month"].unique())
        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        sel_month = st.selectbox("Month", [month_names[m] for m in months], index=len(months)-1)
        sel_month_num = [k for k,v in month_names.items() if v==sel_month][0]
        mdf = df[(df["year"]==sel_year) & (df["month"]==sel_month_num)].copy()

        st.bar_chart(mdf.set_index("date")[["spend","budget"]], height=260)
        cat = mdf.groupby("category")["spend"].sum()
        if not cat.empty:
            st.bar_chart(cat)
            st.pyplot(cat.plot.pie(autopct="%1.1f%%", ylabel="").get_figure())

# --- Yearly Trends ---
elif menu == "Yearly Trends":
    st.subheader("📆 Yearly Trends")
    df = history_df()
    if df.empty:
        st.info("No data yet.")
    else:
        df["dt"] = pd.to_datetime(df["date"])
        yearly = df.groupby(df["dt"].dt.to_period("M")).agg(spend=("spend","sum"), budget=("budget","sum")).reset_index()
        yearly["month"] = yearly["dt"].astype(str)
        st.line_chart(yearly.set_index("month")[["spend","budget"]], height=280)
        ytd = df[df["dt"].dt.year == date.today().year]
        topcats = ytd.groupby("category")["spend"].sum()
        if not topcats.empty:
            st.bar_chart(topcats)
            st.pyplot(topcats.plot.pie(autopct="%1.1f%%", ylabel="").get_figure())

# --- Rewards ---
elif menu == "Rewards":
    st.subheader("🎁 Rewards & Coupons")
    for i, c in enumerate(st.session_state.coupons):
        cols = st.columns([4,2,2])
        with cols[0]:
            st.markdown(f"<div class='coupon'><span class='coin'></span><strong>{c['name']}</strong></div>", unsafe_allow_html=True)
        with cols[1]:
            st.write(f"Cost: **{c['cost']}** coins")
        with cols[2]:
            if c["redeemed"]:
                st.button("Redeemed ✅", key=f"redeemed_{i}", disabled=True)
            else:
                if st.button("Redeem", key=f"redeem_{i}"):
                    redeem_coupon(i)

# --- AI Coach ---
elif menu == "AI Coach":
    st.subheader("🤖 Personal Finance Coach")
    persona = st.selectbox("Answer style", ["student","professional","beginner","expert"], index=0)
    q = st.text_area("Your question", placeholder="e.g., How can I reduce my monthly food spending?")
    if st.button("Ask"):
        if q.strip():
            with st.spinner("Thinking..."):
                ans = ai_answer(q.strip(), persona=persona, model_name=hf_model_choice, max_new_tokens=max_tokens)
            st.write("**Answer:**")
            st.write(ans)

# Reset button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset (clear in-memory data)"):
    st.session_state.clear()
    init_state()
    st.experimental_rerun()
