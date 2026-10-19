import pandas as pd
import json
import ast
import re
import os
from rapidfuzz import fuzz
from datetime import datetime

def evaluate_extraction_detailed(ground_truth_list, model_output_list, fuzzy_threshold=85, output_dir=None):
    total_examples = len(ground_truth_list)
    field_stats = {}
    field_errors_log = {}
    overall_errors_log = [] 
    perfect_matches = 0

    for i in range(total_examples):
        gt = ground_truth_list[i]
        pred = model_output_list[i]
        
        example_is_perfect = True
        example_errors = {}

        all_fields = set(gt.keys()).union(set(pred.keys()))

        for field in all_fields:
            # تجاهل الحقول الداخلية مثل _raw_input
            if field.startswith("_"):
                continue

            gt_value = gt.get(field, None)
            pred_value = pred.get(field, None)
            
            if field not in field_stats:
                field_stats[field] = {"total": 0, "matched": 0, "sum_score": 0.0}
                field_errors_log[field] = []
            
            field_stats[field]["total"] += 1
            
            is_matched = False
            score = 0.0

            # معالجة القيم الفارغة (None)
            if gt_value is None and pred_value is None:
                is_matched = True
                score = 100.0
            elif gt_value is None or pred_value is None:
                is_matched = False
                score = 0.0

            # 1. التطابق التام (العملة، العمولة، أو القيم البوليانية)
            elif field in ["currency", "commission"] or isinstance(gt_value, bool):
                if gt_value == pred_value:
                    is_matched = True
                    score = 100.0
                else:
                    score = 0.0

            # 2. المبالغ المالية (مقارنة رقمية تقبل int و float)
            elif field == "amount":
                try:
                    if float(gt_value) == float(pred_value):
                        is_matched = True
                        score = 100.0
                except (ValueError, TypeError):
                    if str(gt_value).strip() == str(pred_value).strip():
                        is_matched = True
                        score = 100.0

            # 3. أرقام الهواتف (تطابق تام بعد تنظيف المسافات ومحارف اليونيكود الخفية)
            elif field in ["recipientPhone", "phone_number"]:
                clean_gt = re.sub(r'[\s\u200e\u200f\u202a-\u202e]', '', str(gt_value))
                clean_pred = re.sub(r'[\s\u200e\u200f\u202a-\u202e]', '', str(pred_value))
                if clean_gt == clean_pred:
                    is_matched = True
                    score = 100.0
                else:
                    score = 0.0

            # 4. الحقول النصية (الاسم، الغرض) باستخدام التطابق الضبابي
            else:
                norm_gt = str(gt_value or "").strip()
                norm_pred = str(pred_value or "").strip()
                
                score = fuzz.token_sort_ratio(norm_gt, norm_pred)
                if score >= fuzzy_threshold:
                    is_matched = True

            field_stats[field]["sum_score"] += score
            if is_matched:
                field_stats[field]["matched"] += 1
            else:
                example_is_perfect = False
                example_errors[field] = {
                    "expected (GT)": gt_value,
                    "predicted (Model)": pred_value,
                    "match_score": round(score, 2)
                }

                # إضافة العينة التي تحتوي على خطأ إلى سجل هذا الكيان المحدد
                field_errors_log[field].append({
                    "example_index": gt.get("_source_row", i + 2),
                    "input_text": gt.get("_raw_input", ""),
                    "expected": gt_value,
                    "predicted": pred_value,
                    "match_score": round(score, 2),
                    "full_ground_truth": {k: v for k, v in gt.items() if not k.startswith("_")},
                    "full_model_output": pred
                })

        clean_gt = {k: v for k, v in gt.items() if not k.startswith("_")}
        if example_is_perfect:
            perfect_matches += 1
        else:
            overall_errors_log.append({
                "example_index": gt.get("_source_row", i + 2),
                "input_text": gt.get("_raw_input", ""),
                "ground_truth": clean_gt,
                "model_output": pred,
                "field_errors": example_errors
            })

    # --- طباعة التقرير الشامل في الشاشة ---
    print("=" * 60)
    print(f"📊 تقرير التقييم الإجمالي (عدد الأمثلة: {total_examples})")
    print("=" * 60)
    overall_accuracy = (perfect_matches / total_examples) * 100
    print(f"✨ نسبة التطابق التام الكامل للنموذج (Exact Match Rate): {overall_accuracy:.2f}%\n")

    print("--- 📌 أداء كل حقل (Field-level Summary) ---")
    fields_summary = {}
    for field, data in field_stats.items():
        if field.startswith("_"):
            continue
        field_acc = (data["matched"] / data["total"]) * 100
        avg_score = data["sum_score"] / data["total"]
        errors_count = data["total"] - data["matched"]

        fields_summary[field] = {
            "total_examples": data["total"],
            "matched": data["matched"],
            "errors_count": errors_count,
            "success_rate": f"{field_acc:.2f}%",
            "average_match_score": f"{avg_score:.2f}%"
        }

        print(f"• الحقل [{field}]:")
        print(f"  - نسبة النجاح: {field_acc:.2f}% (أخطاء: {errors_count})")
        print(f"  - متوسط جودة التطابق: {avg_score:.2f}%\n")

    print(f"❌ إجمالي الأمثلة التي بها أخطاء: {len(overall_errors_log)} من أصل {total_examples}")

    # --- 📁 إنشاء مجلد جديد لكل تشغيل وحفظ الملفات بداخله ---
    if output_dir is None:
        date_str = datetime.now().strftime("%Y%m%d")
        run_number = 1
        while os.path.exists(f"evaluation-{date_str}-{run_number}"):
            run_number += 1
        output_dir = f"evaluation-{date_str}-{run_number}"

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📂 تم إنشاء المجلد الجديد للتقييم: **{output_dir}/**")

    # 1. ملف نتيجة التقييم بدون تفاصيل (Summary Only)
    summary_data = {
        "evaluation_summary": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_examples": total_examples,
            "perfect_matches": perfect_matches,
            "exact_match_rate": f"{overall_accuracy:.2f}%",
            "total_examples_with_errors": len(overall_errors_log)
        },
        "fields_summary": fields_summary
    }
    summary_file_path = os.path.join(output_dir, "summary.json")
    with open(summary_file_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=4)
    print(f"  📄 تم حفظ الملخص الإجمالي بدون تفاصيل: {summary_file_path}")

    # 2. ملفات منفصلة لكل كيان مع عينات الأخطاء الخاصة به
    for field, errors in field_errors_log.items():
        field_file_path = os.path.join(output_dir, f"{field}.json")
        field_data = {
            "entity": field,
            "summary": fields_summary.get(field, {}),
            "errors_count": len(errors),
            "error_samples": errors
        }
        with open(field_file_path, "w", encoding="utf-8") as f:
            json.dump(field_data, f, ensure_ascii=False, indent=4)
        print(f"  📄 تم حفظ أخطاء الكيان [{field}]: {field_file_path} (عدد الأخطاء: {len(errors)})")

    # 3. ملف الأخطاء الشامل (All Errors) لجميع الكيانات
    if overall_errors_log:
        all_errors_path = os.path.join(output_dir, "all_errors.json")
        all_errors_data = {
            "total_errors": len(overall_errors_log),
            "errors_details": overall_errors_log
        }
        with open(all_errors_path, "w", encoding="utf-8") as f:
            json.dump(all_errors_data, f, ensure_ascii=False, indent=4)
        print(f"  📄 تم حفظ التقرير الشامل للأخطاء: {all_errors_path}")

    return output_dir


# ==========================================
# 🚀 تنفيذ قراءة ملف الـ CSV وتشغيل التقييم
# ==========================================
if __name__ == "__main__":
    file_name = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "najm-dataset-13-10-2026.csv",
    )
    
    try:
        df = pd.read_csv(file_name)
        
        gt_list = []
        pred_list = []
        skipped_rows = 0
        skipped_reasons = {}
        
        for idx, row in df.iterrows():
            parsed_row = {}
            skip_reason = None
            for column in ("output", "mix_output"):
                raw_value = row[column]
                if pd.isna(raw_value):
                    parsed_row = None
                    skip_reason = f"الحقل [{column}] فارغ"
                    break

                raw_value = str(raw_value).strip()
                try:
                    parsed_value = json.loads(raw_value)
                except json.JSONDecodeError:
                    try:
                        parsed_value = ast.literal_eval(raw_value)
                    except (SyntaxError, ValueError):
                        parsed_row = None
                        skip_reason = f"تعذر تحليل الحقل [{column}]"
                        break

                if not isinstance(parsed_value, dict):
                    parsed_row = None
                    skip_reason = f"الحقل [{column}] ليس كائناً"
                    break
                parsed_row[column] = {
                    key: None if value == "None" else value
                    for key, value in parsed_value.items()
                }

            if parsed_row is None:
                skipped_rows += 1
                skipped_reasons[skip_reason] = skipped_reasons.get(skip_reason, 0) + 1
                continue

            parsed_row["output"]["_raw_input"] = row["input"]
            parsed_row["output"]["_source_row"] = idx + 2
            gt_list.append(parsed_row["output"])
            pred_list.append(parsed_row["mix_output"])

        print(f"ℹ️ عدد الصفوف التي تم تخطيها: {skipped_rows}")
        for reason, count in skipped_reasons.items():
            print(f"  - {reason}: {count}")
        
        # تشغيل التقييم (سيتم إنشاء مجلد جديد تلقائياً مثل evaluation-YYYYMMDD-1/)
        evaluate_extraction_detailed(gt_list, pred_list, fuzzy_threshold=85)

    except FileNotFoundError:
        print(f"⚠️ تنبيه: لم يتم العثور على الملف {file_name}. تأكد من وجوده في نفس مسار السكربت.")