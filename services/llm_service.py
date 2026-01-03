import os
import json
import requests
from dotenv import load_dotenv
import dateparser
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
try:
    from pydantic import BaseModel, Field, ValidationError
except Exception:
    # If pydantic not installed, we'll fallback to no-op validation but keep diagnostics
    BaseModel = None
    ValidationError = Exception

load_dotenv()
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
API_KEY_PRESENT = bool(OPENROUTER_KEY)
print(f"[llm_service] OPENROUTER_API_KEY present: {API_KEY_PRESENT}")

def generate_meeting_notes(transcript_text):
    """Generate structured meeting notes using OpenRouter API with professional prompting"""
    if not OPENROUTER_KEY:
        return {
            "summary": "Ringkasan tidak tersedia (API key tidak dikonfigurasi).",
            "highlights": {},
            "actionItems": [],
            "conclusion": "",
            "__llm_diagnostics": {
                "api_key_present": False,
                "fallback": True,
                "note": "OPENROUTER_API_KEY not set in faster-whisper service"
            }
        }
    
    try:
        prompt = """Kamu adalah sekretaris rapat profesional. 
Tugasmu adalah membuat notulensi rapat dalam bahasa Indonesia berdasarkan transkrip berikut.

⚠️ Aturan Penting:
1. Struktur output wajib (Strict JSON):
   - suggestedTitle → Usulan judul rapat yang ringkas dan deskriptif (misal: "Rapat Evaluasi Q3 Marketing").
   - suggestedDescription → Ringkasan 1 kalimat tentang isi rapat max. 20 kata (misal: "Pembahasan target pencapaian Q3 dan alokasi budget marketing.").
   - tags → Daftar 5-8 kata kunci atau topik utama rapat (Array of strings, misal: ["budget", "marketing", "Q3", "evaluasi"]).
   - Summary → ringkasan singkat (2–3 paragraf) dalam format markdown. Gunakan **bold** untuk poin penting.
   - Highlights → catatan detail berdasarkan topik utama dalam format OBJECT (Key-Value Dictionary). JANGAN gunakan Array/List.
       • Key: Sub-judul topik (misal: "Gambaran Umum", "Budget", "Next Steps").
       • Value: Isi catatan detail dalam format markdown (gunakan bullet points `- ` dan **bold**).
       • Contoh salah: "highlights": ["catatan 1", "catatan 2"]
       • Contoh benar: "highlights": { "Topik A": "- Poin 1\\n- Poin 2", "Topik B": "Paragraf penjelasan." }
   - Action Items → daftar tugas (Array of Objects).
       • Wajib: title, description (detail), priority (low/medium/high/urgent).
       • dueDate: PENTING! Ekstrak tanggal deadline dari konteks transkrip. Jika disebutkan tanggal (misal "14 Oktober", "minggu depan", "3 hari lagi"), konversi ke format ISO YYYY-MM-DD. Jika tidak ada tanggal, isi null.
       • Assignee: selalu null. Labels: array string. Status: "todo".
       • Hint: Hanya catat tugas yang jelas dan actionable. Jangan mencatat hal sepele.
   - Conclusion → penutup rapat (Markdown).

2. Gaya Bahasa: Formal, profesional, jelas.
3. Markdown Usage: Gunakan syntax markdown (bold, italic, lists) di dalam string value JSON agar tampilan rapi.
4. JSON Safety: JANGAN gunakan newline literal di dalam string value. Gunakan \\n. Escape double quotes (\") jika ada di dalam string.

Format output JSON HARUS persis seperti ini (CONTOH!):
{
  "suggestedTitle": "Rapat Strategi Pemasaran Q3",
  "suggestedDescription": "Pembahasan rencana peluncuran kampanye media sosial untuk kuartal ketiga.",
  "summary": "**Rapat ini** membahas tentang strategi pemasaran Q3. Disepakati bahwa...",
  "highlights": {
    "Strategi Pemasaran": "- Fokus pada **media sosial**.\\n- Budget dialokasikan sebesar Rp 50jt.",
    "Timeline": "- Peluncuran: **1 Agustus**.\\n- Evaluasi: **15 Agustus**."
  },
  "actionItems": [
    {
      "title": "Siapkan materi kampanye",
      "description": "Buat materi visual untuk Instagram dan TikTok sesuai guideline baru.",
      "assignee": null,
      "priority": "high",
      "dueDate": "2025-08-01",
      "status": "todo",
      "labels": ["marketing", "q3"]
    }
  ],
  "conclusion": "**Kesimpulan rapat**: Kampanye Q3 akan dimulai tepat waktu dengan fokus digital."
}

Hanya kembalikan JSON, tanpa penjelasan tambahan. PASTIKAN semua field diisi dengan konten yang relevan, dan pastikan semuanya digenerate sesuai perintah.

Transkripsi:
""" + transcript_text
        model = os.getenv("LLM_MODEL", "google/gemma-3-27b-it:free")
        referer = os.getenv("APP_REFERER", "http://localhost:3000")
        app_title = os.getenv("APP_TITLE", "Notu.ai")

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": referer,
                "X-Title": app_title,
            },
            data=json.dumps({
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            }),
        )
        
        if response.status_code != 200:
            error_msg = f"OpenRouter Error {response.status_code}: {response.text}"
            print(f"❌ {error_msg}")
            return {
                "summary": f"Error: Gagal generate summary ({response.status_code}).",
                "highlights": {"Error": error_msg},
                "actionItems": [],
                "conclusion": "Gagal menghubungi AI service."
            }
            
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        
        # Robust JSON extraction
        try:
            # 1. Try finding json block
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # 2. Try parsing
            notes_data = json.loads(content)
        except json.JSONDecodeError:
            # Fallback: try to find the outer braces if garbage text exists
            try:
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = content[start:end]
                    # Fix common newlines in string issue by replacing literal newlines with space if not escaped
                    # This is risky but often fixes LLM output
                    json_str = json_str.replace('\n', ' ') 
                    notes_data = json.loads(json_str)
                else:
                    raise ValueError("No JSON object found")
            except Exception as e:
                print(f"❌ JSON Parse Failed. Raw content: {content[:200]}...")
                raise e
        
        # Define simple Pydantic models for validation (if available)
        class _ActionItemModel(BaseModel if BaseModel else object):
            title: Optional[str] = Field(default="Untitled Task")
            description: Optional[str] = Field(default="")
            assignee: Optional[Any] = None
            assigneeName: Optional[str] = None
            priority: Optional[str] = Field(default="medium")
            status: Optional[str] = Field(default="todo")
            labels: Optional[List[str]] = Field(default_factory=list)
            dueDate: Optional[str] = None
            dueDateRaw: Optional[str] = None

        class _MeetingNotesModel(BaseModel if BaseModel else object):
            suggestedTitle: Optional[str] = Field(default="")
            suggestedDescription: Optional[str] = Field(default="")
            tags: Optional[List[str]] = Field(default_factory=list)
            summary: Optional[str] = Field(default="")
            highlights: Optional[Dict[str, str]] = Field(default_factory=dict)
            actionItems: Optional[List[_ActionItemModel]] = Field(default_factory=list)
            conclusion: Optional[str] = Field(default="")

        # Normalize action items: ensure dueDate is ISO (YYYY-MM-DD) or null, and keep dueDateRaw
        raw_action_items = notes_data.get("actionItems", []) if isinstance(notes_data, dict) else []
        normalized_items = []
        validation_errors = None
        try:
            if BaseModel:
                # Validate the whole structure using Pydantic
                _MeetingNotesModel.parse_obj(notes_data)
        except ValidationError as ve:
            validation_errors = str(ve)

        for ai in raw_action_items:
            try:
                title = ai.get("title") if isinstance(ai, dict) else None
            except Exception:
                title = None

            if not isinstance(ai, dict):
                normalized_items.append(ai)
                continue

            due_raw = ai.get("dueDate") if ai.get("dueDate") is not None else ai.get("due_date")
            parsed_iso = None
            if due_raw:
                try:
                    # Try parsing with dateparser using Indonesian locale and additional settings
                    parsed = dateparser.parse(
                        str(due_raw), 
                        languages=['id', 'en'],
                        settings={
                            'PREFER_DATES_FROM': 'future',
                            'RELATIVE_BASE': datetime.now(),
                            'RETURN_AS_TIMEZONE_AWARE': False,
                            'DATE_ORDER': 'DMY'  # Indonesian date format
                        }
                    )
                    if parsed:
                        parsed_iso = parsed.date().isoformat()
                    else:
                        # Fallback: try common Indonesian date patterns manually
                        from datetime import datetime, timedelta
                        import re
                        
                        # Handle relative dates in Indonesian
                        lower_due = str(due_raw).lower().strip()
                        if 'besok' in lower_due:
                            parsed_iso = (datetime.now().date() + timedelta(days=1)).isoformat()
                        elif 'lusa' in lower_due:
                            parsed_iso = (datetime.now().date() + timedelta(days=2)).isoformat()
                        elif 'minggu depan' in lower_due or 'seminggu' in lower_due:
                            parsed_iso = (datetime.now().date() + timedelta(days=7)).isoformat()
                        elif 'bulan depan' in lower_due or 'sebulan' in lower_due:
                            parsed_iso = (datetime.now().date() + timedelta(days=30)).isoformat()
                        elif re.match(r'\d+ (hari|minggu|bulan)', lower_due):
                            # Handle "3 hari", "2 minggu", etc.
                            match = re.match(r'(\d+)\s*(hari|minggu|bulan)', lower_due)
                            if match:
                                num, unit = match.groups()
                                num = int(num)
                                if unit == 'hari':
                                    parsed_iso = (datetime.now().date() + timedelta(days=num)).isoformat()
                                elif unit == 'minggu':
                                    parsed_iso = (datetime.now().date() + timedelta(weeks=num)).isoformat()
                                elif unit == 'bulan':
                                    parsed_iso = (datetime.now().date() + timedelta(days=num*30)).isoformat()
                except Exception:
                    parsed_iso = None

            normalized = {
                "title": ai.get("title", ai.get("text", "Untitled Task")),
                "description": ai.get("description", ""),
                "assignee": ai.get("assignee", None),
                "assigneeName": ai.get("assigneeName") if ai.get("assigneeName") is not None else ai.get("assignee_name"),
                "priority": ai.get("priority", "medium"),
                "status": ai.get("status", "todo"),
                "labels": ai.get("labels", []),
                # Use parsed ISO date if available, otherwise use raw date string (may already be ISO format)
                "dueDate": parsed_iso if parsed_iso else due_raw,
            }
            normalized_items.append(normalized)

        # Ensure structure
        result = {
            "suggestedTitle": notes_data.get("suggestedTitle", ""),
            "suggestedDescription": notes_data.get("suggestedDescription", ""),
            "tags": notes_data.get("tags", []),
            "summary": notes_data.get("summary", ""),
            "highlights": notes_data.get("highlights", {}),
            "actionItems": normalized_items,
            "conclusion": notes_data.get("conclusion", "")
        }
        # attach diagnostics for observability (include validation errors if any)
        result["__llm_diagnostics"] = {
            "api_key_present": API_KEY_PRESENT,
            "fallback": False,
            "parsed_action_items": len(normalized_items),
            "validation_error": validation_errors
        }

        print(f"📋 Generated meeting notes: parsed_action_items={len(normalized_items)} summary_len={len(result['summary'])}")
        return result
    except Exception as e:
        print(f"⚠️ Meeting notes generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "summary": "Ringkasan tidak tersedia.",
            "highlights": {},
                "actionItems": [],
                "conclusion": "",
                "__llm_diagnostics": {
                    "api_key_present": API_KEY_PRESENT,
                    "fallback": True,
                    "error": str(e)
                }
        }
