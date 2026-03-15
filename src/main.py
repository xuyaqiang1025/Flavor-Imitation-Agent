"""
Flavor Imitation Agent - Main Entry Point
===========================================
Complete pipeline: GC-MS -> Parse -> Clean -> Deconvolute -> Validate -> Formula
"""
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from ingestion.parser import parse_gcms_csv, get_summary_stats
from ingestion.cleaner import clean_gcms_data, merge_duplicate_compounds, predict_acetals
from knowledge.inventory_manager import InventoryManager
from knowledge.natural_fingerprints import NaturalFingerprintDB
from engine.deconvoluter import DeconvolutionEngine
from logic.sensory_validator import SensoryValidator
from logic.translator import SensoryTranslator


def analyze_sample(
    gcms_path: str,
    inventory_path: str = None,
    sample_name: str = "Unknown Sample"
) -> dict:
    """
    Run complete analysis pipeline on a GC-MS sample.

    Args:
        gcms_path: Path to GC-MS CSV file.
        inventory_path: Path to inventory CSV (optional).
        sample_name: Name for the analysis.

    Returns:
        Dict with analysis results.
    """
    print(f"\n{'='*60}")
    print(f"  仿香Agent - 配方解析")
    print(f"  样品: {sample_name}")
    print(f"{'='*60}\n")

    # Step 1: Parse GC-MS
    print("[1/6] 解析 GC-MS 数据...")
    df = parse_gcms_csv(gcms_path)
    stats = get_summary_stats(df)
    print(f"      检测到 {stats['total_peaks']} 个峰")

    # Step 2: Predict artifacts
    print("[2/6] 预测反应副产物...")
    acetals = predict_acetals(df)
    if acetals:
        print(f"      发现 {len(acetals)} 个可能的缩醛类副产物")
        for a in acetals:
            print(f"        - {a['reason']}")

    # Step 3: Clean data
    print("[3/6] 清洗数据 (去除杂质/溶剂/尼古丁)...")
    df_clean = clean_gcms_data(df)
    df_merged = merge_duplicate_compounds(df_clean)
    print(f"      清洗后剩余 {len(df_merged)} 个有效化合物")

    # Step 4: Inventory matching (if available)
    if inventory_path:
        print("[4/6] 匹配原料库...")
        inv_mgr = InventoryManager(inventory_path)
        df_matched = inv_mgr.match_gcms_to_inventory(df_merged)
        matched_count = df_matched['in_inventory'].sum()
        print(f"      {matched_count}/{len(df_matched)} 个化合物在库存中")
    else:
        print("[4/6] 跳过原料库匹配 (未提供库存文件)")
        df_matched = df_merged

    # Step 5: Deconvolution
    print("[5/6] 运行解卷积算法...")
    engine = DeconvolutionEngine()
    result = engine.analyze(df_matched)

    print("\n      推理过程:")
    for r in result.reasoning:
        print(f"        {r}")

    print("\n      识别到的天然提取物:")
    if result.naturals:
        for name, pct in result.naturals.items():
            print(f"        • {name}: {pct:.1f}%")
    else:
        print("        (无明显天然提取物，可能为纯合成配方)")

    # Step 6: Generate formula
    print("\n[6/6] 生成配方...")
    formula = engine.generate_formula(result, df_matched)

    # Validate
    validator = SensoryValidator()
    issues = validator.validate_formula(formula)

    print("\n" + "="*60)
    print("  生成配方")
    print("="*60)
    print(formula.to_string(index=False))

    if issues:
        print("\n⚠ 验证警告:")
        for issue in issues:
            print(f"  [{issue.severity}] {issue.ingredient}: {issue.message}")

    if result.warnings:
        print("\n⚠ 分析警告:")
        for w in result.warnings:
            print(f"  {w}")

    print(f"\n置信度: {result.confidence:.1f}%")
    print("="*60 + "\n")

    return {
        'raw_peaks': stats['total_peaks'],
        'clean_compounds': len(df_merged),
        'naturals': result.naturals,
        'synthetics_count': len(result.synthetics),
        'formula': formula,
        'confidence': result.confidence,
        'issues': issues
    }


def interactive_feedback(formula_df, feedback: str) -> None:
    """
    Process user sensory feedback and suggest adjustments.
    """
    print(f"\n用户反馈: '{feedback}'")
    print("-" * 40)

    # Create translator with current formula
    formula_dict = dict(zip(formula_df['Name'], formula_df['Percentage']))
    translator = SensoryTranslator(formula_dict)

    adjustments = translator.translate(feedback)

    if adjustments:
        print("建议调整:")
        for adj in adjustments:
            print(f"  {translator.explain_adjustment(adj)}")

        # Apply and show new formula
        new_formula = translator.apply_adjustments(formula_df, adjustments)
        print("\n调整后配方 (前5项):")
        print(new_formula.head().to_string(index=False))
    else:
        print("未能解析反馈，请尝试使用更具体的描述词，如:")
        print("  '太甜'、'不够凉'、'香草味淡'、'有刺激感' 等")


if __name__ == '__main__':
    # Run analysis on sample file
    sample_gcms = Path(__file__).parent.parent / 'data' / 'raw_gcms' / 'Sample1.csv'
    sample_inventory = Path(__file__).parent.parent / 'data' / 'inventory' / 'sample_inventory.csv'

    if sample_gcms.exists():
        results = analyze_sample(
            gcms_path=str(sample_gcms),
            inventory_path=str(sample_inventory) if sample_inventory.exists() else None,
            sample_name="DC弗吉尼亚烟草烟油"
        )

        # Demo: Process feedback
        print("\n" + "="*60)
        print("  交互式反馈演示")
        print("="*60)
        interactive_feedback(results['formula'], "太甜了，需要更凉一点")
    else:
        print(f"Sample file not found: {sample_gcms}")
        print("Please place a GC-MS CSV file in data/raw_gcms/")
