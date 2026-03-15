"""인삼판매관리 Excel 파일 분석 스크립트"""
import os
import xlrd
import json

FOLDER = r"C:\Jinseng"

results = {}

for fname in sorted(os.listdir(FOLDER)):
    if not fname.lower().endswith(('.xls', '.xlsx')):
        continue
    fpath = os.path.join(FOLDER, fname)
    print(f"\n{'='*80}")
    print(f"파일: {fname}  (크기: {os.path.getsize(fpath):,} bytes)")
    print('='*80)
    
    try:
        wb = xlrd.open_workbook(fpath)
    except Exception as e:
        print(f"  [오류] {e}")
        results[fname] = {"error": str(e)}
        continue
    
    file_info = {"sheets": []}
    
    for sname in wb.sheet_names():
        sh = wb.sheet_by_name(sname)
        print(f"\n  시트: '{sname}'  ({sh.nrows}행 x {sh.ncols}열)")
        
        sheet_info = {
            "name": sname,
            "rows": sh.nrows,
            "cols": sh.ncols,
            "header_rows": [],
            "sample_rows": []
        }
        
        # 처음 10행 출력 (헤더 + 샘플 데이터)
        max_preview = min(15, sh.nrows)
        for r in range(max_preview):
            row_vals = []
            for c in range(sh.ncols):
                cell = sh.cell(r, c)
                if cell.ctype == xlrd.XL_CELL_DATE:
                    try:
                        dt = xlrd.xldate_as_tuple(cell.value, wb.datemode)
                        row_vals.append(f"{dt[0]}-{dt[1]:02d}-{dt[2]:02d}")
                    except:
                        row_vals.append(str(cell.value))
                elif cell.ctype == xlrd.XL_CELL_NUMBER:
                    if cell.value == int(cell.value):
                        row_vals.append(str(int(cell.value)))
                    else:
                        row_vals.append(str(cell.value))
                else:
                    row_vals.append(str(cell.value).strip())
            
            row_str = " | ".join(row_vals)
            print(f"    [{r:3d}] {row_str}")
            
            if r < 3:
                sheet_info["header_rows"].append(row_vals)
            else:
                sheet_info["sample_rows"].append(row_vals)
        
        # 마지막 3행도 출력 (데이터 끝부분 확인)
        if sh.nrows > max_preview:
            print(f"    ... (중간 {sh.nrows - max_preview - 3}행 생략) ...")
            for r in range(max(max_preview, sh.nrows - 3), sh.nrows):
                row_vals = []
                for c in range(sh.ncols):
                    cell = sh.cell(r, c)
                    if cell.ctype == xlrd.XL_CELL_DATE:
                        try:
                            dt = xlrd.xldate_as_tuple(cell.value, wb.datemode)
                            row_vals.append(f"{dt[0]}-{dt[1]:02d}-{dt[2]:02d}")
                        except:
                            row_vals.append(str(cell.value))
                    elif cell.ctype == xlrd.XL_CELL_NUMBER:
                        if cell.value == int(cell.value):
                            row_vals.append(str(int(cell.value)))
                        else:
                            row_vals.append(str(cell.value))
                    else:
                        row_vals.append(str(cell.value).strip())
                row_str = " | ".join(row_vals)
                print(f"    [{r:3d}] {row_str}")
        
        # 병합 셀 정보
        if hasattr(sh, 'merged_cells') and sh.merged_cells:
            print(f"    [병합 셀] {len(sh.merged_cells)}개")
            for mc in sh.merged_cells[:10]:
                print(f"      ({mc[0]},{mc[2]}) ~ ({mc[1]-1},{mc[3]-1})")
        
        file_info["sheets"].append(sheet_info)
    
    results[fname] = file_info

print("\n\n" + "="*80)
print("요약")
print("="*80)
for fname, info in results.items():
    if "error" in info:
        print(f"  {fname}: 오류 - {info['error']}")
    else:
        for s in info["sheets"]:
            print(f"  {fname} / '{s['name']}': {s['rows']}행 x {s['cols']}열")
