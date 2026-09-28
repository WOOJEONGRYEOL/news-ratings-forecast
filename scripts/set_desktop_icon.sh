#!/usr/bin/env bash
# 바탕화면 런처에 '평평한' 커스텀 아이콘을 지정한다.
#
# macOS 26 은 .app 번들 아이콘에 규격 컨테이너(둥근 마스크 + 밝은 테두리 링)를
# 덧씌운다. 파일의 커스텀 아이콘에는 덧씌우지 않으므로, 이 방식으로 지정해야
# 다른 런처(TTS Studio.command 등)와 테두리 없이 같은 크기로 보인다.
#
# 아이콘을 다시 만든 뒤에는 반드시 이 스크립트를 실행할 것.
set -euo pipefail

PNG="${1:-/Users/woo/FORECAST/assets/icon_1024.png}"
APP="${2:-$HOME/Desktop/NEWSCAST.app}"

[ -f "$PNG" ] || { echo "PNG 없음: $PNG" >&2; exit 1; }
[ -d "$APP" ] || { echo "앱 없음: $APP" >&2; exit 1; }

osascript -l JavaScript -e "
ObjC.import('AppKit');
var img = \$.NSImage.alloc.initWithContentsOfFile('$PNG');
if (!\$.NSWorkspace.sharedWorkspace.setIconForFileOptions(img, '$APP', 0))
  throw new Error('아이콘 지정 실패');
"
killall Finder 2>/dev/null || true
echo "적용: $(basename "$APP") ← $(basename "$PNG")"
