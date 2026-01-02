"""
Chat routes for Ask AI feature
Handles LLM chat completions for meeting Q&A
"""
from flask import Blueprint, request, jsonify
import os
import json
import requests

chat_bp = Blueprint('chat', __name__)

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

@chat_bp.route('/api/chat', methods=['POST'])
def chat_completion():
    """
    Chat completion endpoint for Ask AI feature
    Expects: { messages: [{ role, content }], max_tokens?: number }
    Returns: { success: true, response: string }
    """
    try:
        data = request.get_json()
        
        if not data or 'messages' not in data:
            return jsonify({
                'success': False,
                'error': 'messages array is required'
            }), 400
        
        messages = data.get('messages', [])
        max_tokens = data.get('max_tokens', 1000)
        
        if not OPENROUTER_KEY:
            return jsonify({
                'success': False,
                'error': 'OPENROUTER_API_KEY not configured',
                'response': 'Maaf, layanan AI belum dikonfigurasi. Silakan hubungi administrator.'
            }), 503
        
        # Call OpenRouter
        model = os.getenv("LLM_MODEL", "google/gemini-2.0-flash-exp:free")
        referer = os.getenv("APP_REFERER", "http://localhost:3000")
        app_title = os.getenv("APP_TITLE", "Notu.ai")
        
        # Fix for Gemma models: They don't support system messages
        # Convert system message to user message by prepending to first user message
        processed_messages = []
        system_content = None
        
        for msg in messages:
            if msg.get('role') == 'system':
                system_content = msg.get('content', '')
            else:
                if system_content and msg.get('role') == 'user' and not processed_messages:
                    # Prepend system content to first user message
                    processed_messages.append({
                        'role': 'user',
                        'content': f"{system_content}\n\n{msg.get('content', '')}"
                    })
                    system_content = None  # Clear after using
                else:
                    processed_messages.append(msg)
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": referer,
                "X-Title": app_title,
            },
            json={
                "model": model,
                "messages": processed_messages,
                "max_tokens": max_tokens,
            },
            timeout=60
        )
        
        if response.status_code != 200:
            error_msg = f"OpenRouter Error {response.status_code}"
            print(f"❌ {error_msg}: {response.text}")
            return jsonify({
                'success': False,
                'error': error_msg,
                'response': 'Maaf, terjadi kesalahan saat menghubungi layanan AI.'
            }), response.status_code
        
        result = response.json()
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        if not content:
            return jsonify({
                'success': False,
                'error': 'Empty response from LLM',
                'response': 'Maaf, AI tidak dapat memberikan jawaban. Silakan coba lagi.'
            }), 500
        
        return jsonify({
            'success': True,
            'response': content,
            'model': model,
            'usage': result.get('usage', {})
        })
        
    except requests.Timeout:
        return jsonify({
            'success': False,
            'error': 'Request timeout',
            'response': 'Maaf, permintaan memakan waktu terlalu lama. Silakan coba lagi.'
        }), 504
        
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'response': 'Maaf, terjadi kesalahan. Silakan coba lagi.'
        }), 500
