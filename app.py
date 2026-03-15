"""
Flavor Imitation Agent - Interactive UI Dashboard (Streamlit)
===============================================================
A professional analysis dashboard for flavor engineers.
Features: GC-MS upload, Formula Editing, Interactive Charts, Feedback Loop.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ingestion.parser import parse_gcms_csv
from ingestion.cleaner import clean_gcms_data, merge_duplicate_compounds, predict_acetals
from knowledge.inventory_manager import InventoryManager
from engine.deconvoluter import DeconvolutionEngine
from logic.sensory_validator import SensoryValidator
from logic.translator import SensoryTranslator
from logic.llm_agent import FlavorLLMAgent  # NEW Import

# Page Config
st.set_page_config(
    page_title="仿香 Agent Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6 }
    .main .block-container { padding-top: 2rem; }
    h1 { color: #1e3a8a; }
    h2 { color: #1e40af; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_resources():
    """Load heavy resources like inventory database."""
    # Use path relative to this script file
    base_dir = Path(__file__).parent
    inventory_path = base_dir / "data" / "inventory" / "sample_inventory.csv"
    
    if not inventory_path.exists():
        # Fallback: try User's actual inventory if named 'inventory.csv'
        user_inv = base_dir / "data" / "inventory" / "inventory.csv"
        if user_inv.exists():
            inventory_path = user_inv
        else:
            st.warning(f"Inventory file not found at {inventory_path}")
            return None
            
    return InventoryManager(inventory_path)


def main():
    st.title("🧪 仿香 Agent | 智能配方反汇编系统")
    st.caption("电子烟香精配方逆向工程与自动化辅助设计工具 - Agentic Edition")

    # Sidebar: Setup
    with st.sidebar:
        st.header("1. 数据输入")
        uploaded_file = st.file_uploader("上传 GC-MS 数据 (CSV)", type=["csv"])
        
        st.header("2. AI 配置 (New)")
        api_provider = st.selectbox("LLM模型服务商", ["OpenAI", "DeepSeek", "自定义"], index=0)
        api_key = st.text_input("API Key", type="password", help="输入您的API Key以启用智能推理")
        base_url = st.text_input("Base URL (可选)", value="https://api.openai.com/v1" if api_provider=="OpenAI" else "https://api.deepseek.com/v1")
        
        st.header("3. 参数配置")
        max_naturals = st.slider("最大天然物识别数", 1, 10, 5)
        clean_solvents = st.checkbox("自动去除溶剂 (PG/VG)", value=True)
        
        st.header("4. 知识库Status")
        inv_mgr = load_resources()
        if inv_mgr:
            st.success(f"✅ 原料库: {len(inv_mgr.inventory_df)} 条")
        else:
            st.error("❌ 原料库未加载")

        st.markdown("---")
        st.info("💡 提示: 配置 API Key 可开启 LLM 智能调香功能")

    # Main Area
    if uploaded_file is not None:
        try:
            # 1. Parse & Process
            with st.spinner("正在解析图谱..."):
                # Save uploaded file temporarily to parse
                with open("temp_upload.csv", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                df_raw = parse_gcms_csv("temp_upload.csv")
                df_clean = clean_gcms_data(
                    df_raw, 
                    remove_solvents=clean_solvents,
                    remove_nicotine=True
                )
                df_merged = merge_duplicate_compounds(df_clean)

                # Match Inventory
                if inv_mgr:
                    df_matched = inv_mgr.match_gcms_to_inventory(df_merged)
                else:
                    df_matched = df_merged

            # 2. Deconvolution Analysis
            with st.spinner("正在进行智能解卷积..."):
                engine = DeconvolutionEngine()
                result = engine.analyze(df_matched, max_naturals=max_naturals)
                formula_df = engine.generate_formula(result, df_matched)

            # --- DASHBOARD LAYOUT ---
            
            # Top Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("原始峰数量", len(df_raw))
            col2.metric("有效化合物", len(df_merged))
            col3.metric("识别天然物", len(result.naturals))
            col4.metric("解析置信度", f"{result.confidence:.1f}%")

            # Tabs
            tab1, tab2, tab3 = st.tabs(["📊 配方分析", "🧠 智能调香师 (LLM)", "📑 原始数据"])

            with tab1:
                st.subheader("生成配方建议")
                
                # Split view: Chart and Table
                c1, c2 = st.columns([1, 2])
                
                with c1:
                    # Pie Chart of Types
                    type_dist = formula_df.groupby('Type')['Percentage'].sum().reset_index()
                    fig_pie = px.pie(type_dist, values='Percentage', names='Type', 
                                    title='天然 vs 合成 比例', hole=0.4,
                                    color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig_pie, use_container_width=True)

                    # Naturals list
                    if result.naturals:
                        st.markdown("### 🌿 识别到的天然提取物")
                        for name, pct in result.naturals.items():
                            st.success(f"**{name}**: {pct:.2f}%")
                    else:
                        st.info("未识别到明显的天然提取物")

                with c2:
                    # Interactive Formula Table
                    st.markdown("### 📋 初始配方表 (可编辑)")
                    edited_formula = st.data_editor(
                        formula_df[['Name', 'Percentage', 'Type', 'Role', 'Notes']],
                        num_rows="dynamic",
                        use_container_width=True,
                        height=400,
                        key="formula_editor"
                    )
                    
                    total_pct = edited_formula['Percentage'].sum()
                    if total_pct < 95 or total_pct > 105:
                        st.warning(f"⚠️ 当前配方总和: {total_pct:.1f}% (建议调整至 100%)")
                    else:
                        st.success(f"✅ 当前配方总和: {total_pct:.1f}%")

            with tab2:
                st.subheader("🧠 智能调香师 (LLM-Powered)")
                
                if not api_key:
                    st.warning("⚠️ 请在左侧侧边栏配置 API Key 以启用智能功能")
                
                col_feed, col_res = st.columns([1, 1])
                
                with col_feed:
                    st.markdown("输入您的感官评价，AI Agent将基于化学原理进行推理：")
                    feedback_text = st.text_area("感官反馈 (例如: '太甜了且中段空洞，想要那种类似万宝路的干草味')", height=100)
                    
                    if st.button("🤖 询问 AI Agent"):
                        if feedback_text:
                            # Use New LLM Agent
                            with st.spinner("AI 正在思考化学调性..."):
                                formula_dict = edited_formula.to_dict('records')
                                inventory_context = f"Available Inventory Size: {len(inv_mgr.inventory_df)}" if inv_mgr else "No Inventory Loaded"
                                
                                agent = FlavorLLMAgent(api_key=api_key, base_url=base_url)
                                adjustments = agent.suggest_adjustments(feedback_text, formula_dict, inventory_context)
                                
                                st.session_state['llm_adjustments'] = adjustments
                        else:
                            st.warning("请输入反馈内容")

                with col_res:
                    if 'llm_adjustments' in st.session_state:
                        st.markdown("### 💡 AI 建议调整")
                        adj_list = st.session_state['llm_adjustments']
                        
                        for adj in adj_list:
                            # Handle error case
                            if adj.get('action') == 'Error':
                                st.error(adj.get('reason'))
                                continue
                                
                            emoji = "📈" if adj['action'] in ['increase', 'add'] else "📉"
                            amt_str = f"{adj['amount']}%" if adj['amount'] else "N/A"
                            
                            with st.expander(f"{emoji} {adj['action'].upper()}: {adj['target']} ({amt_str})", expanded=True):
                                st.markdown(f"**Reason**: {adj['reason']}")

                st.markdown("---")
                st.subheader("⚠️ 雾化兼容性检查")
                validator = SensoryValidator(inv_mgr.inventory_df if inv_mgr else None)
                issues = validator.validate_formula(edited_formula)
                
                if issues:
                    for issue in issues:
                        color = "red" if issue.severity == "error" else "orange"
                        st.markdown(f":{color}[[{issue.severity.upper()}]] **{issue.ingredient}**: {issue.message}")
                else:
                    st.success("未发现明显的雾化兼容性问题")

            with tab3:
                st.subheader("原始 GC-MS 数据")
                st.dataframe(df_raw, use_container_width=True)
                
                if result.warnings:
                    st.subheader("分析过程警告")
                    for w in result.warnings:
                        st.warning(w)

        except Exception as e:
            st.error(f"分析出错: {str(e)}")
            st.exception(e)

    else:
        # Welcome Screen
        st.markdown("""
        ### 👋 欢迎使用
        
        请在左侧侧边栏上传您的 GC-MS 数据文件 (.csv) 开始分析。
        
        **功能特性:**
        *   🤖 **自动去噪**: 智能剔除溶剂和杂质
        *   🌿 **天然物拆解**: 自动识别配方中的天然精油
        *   ☁️ **雾化验证**: 确保配方适合电子烟雾化
        *   🗣️ **自然语言微调**: 告诉Agent "太甜"，它会自动帮你改配方
        """)

        # Show Demo Data Button (Optional)
        # if st.button("加载示例数据"):
            # Load demo logic here...

if __name__ == "__main__":
    main()
