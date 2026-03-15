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

# 1. Custom Visual System (Premium Aesthetic)
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* Main App Layout */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Header Styling */
    h1 { 
        color: #0f172a !important; 
        font-weight: 800 !important; 
        transition: all 0.3s ease;
        letter-spacing: -0.025em;
    }
    h2, h3 { 
        color: #334155 !important; 
        font-weight: 700 !important;
        margin-top: 1.5rem !important;
    }
    
    /* Metrics panel styling with glassmorphism */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(226, 232, 240, 0.5) !important;
        padding: 20px !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05) !important;
        transition: transform 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent !important;
        border-radius: 0px;
        font-weight: 400;
        color: #64748b;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #2563eb;
    }
    .stTabs [aria-selected="true"] {
        font-weight: 600 !important;
        color: #2563eb !important;
        border-bottom-color: #2563eb !important;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px !important;
        background-color: #2563eb !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.5rem 1.5rem !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_resources():
    """Load heavy resources like inventory database and Vector DB."""
    # Ensure Vector DB is initialized
    from knowledge.vector_db_builder import VectorDBBuilder
    vdb = VectorDBBuilder()
    if vdb.is_available():
        # Check if empty
        if vdb.collection.count() == 0:
             vdb.build_from_builtin()
             
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
    # 2. Hero Section
    st.markdown("""
    <div style='background: white; padding: 2.5rem; border-radius: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); margin-bottom: 2rem; border: 1px solid #e2e8f0;'>
        <h1 style='margin:0; font-size: 2.5rem;'>🧪 仿香 Agent | <span style='color: #2563eb;'>智能配方反汇编系统</span></h1>
        <p style='color: #64748b; font-size: 1.1rem; margin-top: 0.5rem;'>电子烟香精配方逆向工程与自动化辅助设计工具 - Agentic Edition</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Setup
    with st.sidebar:
        st.header("1. 数据输入")
        uploaded_file = st.file_uploader("上传 GC-MS 数据 (CSV)", type=["csv"])
        st.header("⚙️ 2. AI 智能配置")
        
        # Extended LLM Providers mapping
        PROVIDER_URLS = {
            "DeepSeek (推荐)": "https://api.deepseek.com/v1",
            "OpenAI (GPT-4)": "https://api.openai.com/v1",
            "Anthropic (Claude通过兼容代理)": "https://api.anthropic.com/v1",
            "Google (Gemini通过兼容代理)": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "阿里云 (Qwen)": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "硅基流动 (SiliconFlow)": "https://api.siliconflow.cn/v1",
            "自定义 (Custom)": ""
        }
        
        api_provider = st.selectbox("LLM 模型服务商", list(PROVIDER_URLS.keys()), index=0)
        api_key = st.text_input("🔑 API Key", type="password", help="输入您的API Key以启用智能推理与解卷积")
        
        # Auto-fill Base URL based on selection
        default_url = PROVIDER_URLS.get(api_provider, "")
        base_url = st.text_input("🔗 Base URL (兼容 OpenAI 接口)", value=default_url)
        
        st.header("🎛️ 3. 解析参数配置")
        max_naturals = st.slider("🌿 最大天然物识别数", min_value=1, max_value=10, value=5, help="限制解卷积算法最多识别几种天然提取物")
        clean_solvents = st.checkbox("🧼 自动去除溶剂 (PG/VG)", value=True, help="在指纹分析前剔除大剂量的丙二醇/甘油")
        
        st.divider()
        st.header("📚 4. 本地知识库状态")
        inv_mgr = load_resources()
        if inv_mgr:
            st.success(f"✅ 合成原料库: **{len(inv_mgr.inventory_df)}** 条记录")
        else:
            st.error("❌ 合成原料库未加载")

        # Vector DB status and batch import section
        from knowledge.vector_db_builder import VectorDBBuilder
        _vdb = VectorDBBuilder()
        if _vdb.is_available():
            db_count = _vdb.count()
            st.info(f"🧬 天然指纹库: {db_count} 条 (ChromaDB)")
        else:
            st.warning("❗ ChromaDB 不可用")

        with st.expander("📥 导入自定义指纹数据"):
            st.markdown("支持 **CSV / Excel / Markdown** 格式。[下载模板](data/inventory/my_extracts_template.csv)")
            import_file = st.file_uploader(
                "上传精油指纹文件",
                type=["csv", "xlsx", "md"],
                key="import_fingerprints"
            )
            if import_file is not None and st.button("🔄 导入并更新向量库"):
                with st.spinner("正在解析并写入 ChromaDB..."):
                    suffix = Path(import_file.name).suffix.lower()
                    tmp_path = Path("temp_import") .with_suffix(suffix)
                    tmp_path.write_bytes(import_file.read())
                    try:
                        vdb = VectorDBBuilder()
                        if suffix == ".csv":
                            n = vdb.import_from_csv(str(tmp_path))
                        elif suffix in (".xlsx", ".xls"):
                            n = vdb.import_from_excel(str(tmp_path))
                        elif suffix == ".md":
                            n = vdb.import_from_markdown(str(tmp_path))
                        else:
                            n = 0
                        st.success(f"✅ 成功导入 {n} 条记录，库共 {vdb.count()} 条。")
                        st.cache_resource.clear()  # Force reload
                    except Exception as e:
                        st.error(f"导入失败: {e}")
                    finally:
                        tmp_path.unlink(missing_ok=True)

        st.markdown("---")
        st.info("💡 提示: 配置 API Key 可开启 LLM 智能调香功能")

    # Main Area
    if uploaded_file is not None:
        try:
            # 1. Parse & Process
            with st.spinner("⏳ 正在进行高精度化学图谱解析..."):
                # Save uploaded file temporarily to parse
                with open("temp_upload.csv", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                df_raw = parse_gcms_csv("temp_upload.csv")
                # predict_acetals is usually called inside cleaner or manually here
                # Our new clean_gcms_data does NOT call predict_acetals automatically inside it
                # So we will extract and display if necessary
                predictions = predict_acetals(df_raw, api_key=api_key, base_url=base_url)
                if len(predictions) > 0:
                     st.toast(f"Virtual Reactor predicted {len(predictions)} byproducts to exclude!")

                df_clean = clean_gcms_data(
                    df_raw, 
                    remove_solvents=clean_solvents,
                    remove_nicotine=True
                )
                
                # Dynamic exclusion of LLM-predicted acetals
                if predictions and 'cas' in df_clean.columns:
                     # For now, just exclude by name if CAS isn't reliable from LLM
                     # but clean_gcms_data works by blocklist. 
                     # To be fully dynamic, we filter them manually:
                     predicted_names = [p['predicted_acetal_name'].lower() for p in predictions if 'predicted_acetal_name' in p]
                     if 'compound_name_cn' in df_clean.columns:
                          df_clean = df_clean[~df_clean['compound_name_cn'].str.lower().isin(predicted_names)]
                
                df_merged = merge_duplicate_compounds(df_clean)

                # Match Inventory
                if inv_mgr:
                    df_matched = inv_mgr.match_gcms_to_inventory(df_merged)
                else:
                    df_matched = df_merged

            # 2. Deconvolution Analysis
            with st.spinner("🧠 AI 混合解卷积引擎运转中 (Traceback -> CAS Similarities -> LLM Qualification -> Math Quant)..."):
                engine = DeconvolutionEngine(api_key=api_key if api_key else None, base_url=base_url if base_url else None)
                result = engine.analyze(df_matched, max_naturals=max_naturals)
                formula_df = engine.generate_formula(result, df_matched)

            # --- DASHBOARD LAYOUT ---
            st.markdown("### 📈 快速洞察 (Quick Insights)")
            
            # Top Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📍 原始峰数量", len(df_raw))
            col2.metric("🎯 有效香原料", len(df_merged))
            col3.metric("🌿 天然物发现", len(result.naturals))
            
            # Color code confidence
            conf_color = "green" if result.confidence > 80 else "orange" if result.confidence > 50 else "red"
            col4.metric(f"🔬 解析置信度", f"{result.confidence:.1f}%")

            # Reasoning Trace Expander
            with st.expander("🛠️ 查看 AI 解析推理日志 (Detailed Reasoning Pipeline)", expanded=False):
                if result.reasoning:
                    for line in result.reasoning:
                        # Add some visual grouping based on common keywords
                        if "Step" in line or "Processing" in line:
                            st.markdown(f"#### {line}")
                        elif "Confirmed" in line or "Found" in line:
                            st.success(line)
                        elif "Traceback" in line or "Match" in line:
                            st.info(line)
                        else:
                            st.text(line)
                else:
                    st.info("无推理日志")
                
                if result.warnings:
                    st.divider()
                    st.markdown("**⚠️ 异常/预警信息:**")
                    for w in result.warnings:
                        st.warning(w)

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
                    
                    # Search and Filter for the large table
                    search_term = st.text_input("🔍 搜索成分 (CAS/名称)", "")
                    display_df = formula_df
                    if search_term:
                        display_df = formula_df[
                            formula_df['Name'].str.contains(search_term, case=False, na=False) |
                            formula_df['Notes'].str.contains(search_term, case=False, na=False)
                        ]

                    edited_formula = st.data_editor(
                        display_df[['Name', 'Percentage', 'Type', 'Role', 'Notes']],
                        num_rows="dynamic",
                        use_container_width=True,
                        height=400,
                        key="formula_editor"
                    )
                    
                    # Download Section
                    st.markdown("---")
                    d_col1, d_col2 = st.columns(2)
                    
                    csv = edited_formula.to_csv(index=False).encode('utf-8-sig') # with BOM for Excel
                    d_col1.download_button(
                        label="📥 下载 CSV 配方",
                        data=csv,
                        file_name="suggested_formula.csv",
                        mime="text/csv",
                    )
                    
                    try:
                        import io
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            edited_formula.to_excel(writer, index=False, sheet_name='Formula')
                        excel_data = output.getvalue()
                        d_col2.download_button(
                            label="📄 下载 Excel 配方",
                            data=excel_data,
                            file_name="suggested_formula.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except ImportError:
                        d_col2.info("安装 openpyxl 以启用 Excel 下载")

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
        <div style='text-align: center; padding: 3rem 0;'>
            <h1 style='font-size: 3rem; color: #1e293b; margin-bottom: 1rem;'>🧪 欢迎使用 Flavor Imitation Agent</h1>
            <p style='font-size: 1.25rem; color: #64748b; max-width: 600px; margin: 0 auto 2rem auto;'>
                您的高阶 AI 调香助理。上传 GC-MS 图谱，揭开复杂天然混合物与合成单体的神秘面纱。
            </p>
        </div>
        
        <div style='display: flex; gap: 2rem; justify-content: center; margin-bottom: 2rem;'>
            <div style='background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); flex: 1; max-width: 300px;'>
                <h3 style='margin-top:0;'>🤖 混合推理解卷积</h3>
                <p style='color: #64748b;'>结合 LLM 定性与 CAS 稀疏向量相似度算法，精准剥离精油与单体。</p>
            </div>
            <div style='background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); flex: 1; max-width: 300px;'>
                <h3 style='margin-top:0;'>🧪 虚拟化学反应器</h3>
                <p style='color: #64748b;'>自动监测并溯源陈化过程中的缩醛、缩酮反应副产物，还原配方真相。</p>
            </div>
            <div style='background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); flex: 1; max-width: 300px;'>
                <h3 style='margin-top:0;'>☁️ 雾化适配性拦截</h3>
                <p style='color: #64748b;'>基于分子量、沸点等物理参数，提前预警雾化不全或挂壁积碳风险。</p>
            </div>
        </div>
        
        <div style='text-align: center; color: #94a3b8;'>
            👉 请在左侧侧边栏上传您的 <b>GC-MS 数据文件 (.csv)</b> 以启动旅程。
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
