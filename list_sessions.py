# -*- coding: utf-8 -*-
import json
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SESSION_DIR = r"C:\Users\31228\.claude\projects\C--Users-31228"

def get_first_user_message(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get('type') == 'user':
                    msg = obj.get('message', {})
                    content = msg.get('content', '')
                    if isinstance(content, str) and not content.startswith('<local-command-caveat>') and len(content) > 5 and not content.startswith('<command-name>'):
                        return content[:80]
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                text = item.get('text', '')
                                if text and len(text) > 5:
                                    return text[:80]
    except Exception as e:
        return f"[读取错误: {e}]"
    return "[无内容]"

def main():
    print("=" * 70)
    print(f"{'会话ID':<40} {'首条消息'}")
    print("=" * 70)

    sessions = []
    for filename in os.listdir(SESSION_DIR):
        if filename.endswith('.jsonl'):
            session_id = filename.replace('.jsonl', '')
            filepath = os.path.join(SESSION_DIR, filename)
            first_msg = get_first_user_message(filepath)
            sessions.append((session_id, first_msg))

    # Sort by modification time (newest first)
    sessions.sort(key=lambda x: os.path.getmtime(os.path.join(SESSION_DIR, x[0] + '.jsonl')), reverse=True)

    for i, (sid, msg) in enumerate(sessions, 1):
        print(f"{i:2}. {sid:<38} {msg}")

    print("\n" + "=" * 70)
    print(f"共 {len(sessions)} 个会话")

if __name__ == "__main__":
    main()
