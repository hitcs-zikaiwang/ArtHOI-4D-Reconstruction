import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import json
from datetime import datetime
import argparse
from loguru import logger as loguru

DEVICE = torch.device("cuda")

def parse_ranges(input_str, img_files):
    idx_set = set()
    if input_str.lower() == 'done':
        return idx_set
    ranges = input_str.split(',')
    img_nums = [int(os.path.splitext(f)[0]) for f in img_files]
    num2idx = {num: idx for idx, num in enumerate(img_nums)}
    for r in ranges:
        if '-' not in r:
            continue
        start_num, end_num = map(int, r.split('-'))
        for num in range(start_num, end_num + 1):
            if num in num2idx:
                idx_set.add(num2idx[num])
    return idx_set


def idx_set_to_segments(idx_set: set) -> list:
    if not idx_set:
        return []
    sorted_idx = sorted(idx_set)
    segments = []
    seg_start = sorted_idx[0]
    prev = sorted_idx[0]
    for cur in sorted_idx[1:]:
        if cur == prev + 1:
            prev = cur
            continue
        segments.append((seg_start, prev))
        seg_start = cur
        prev = cur
    segments.append((seg_start, prev))
    return segments

FINGER_ALIASES = {
    'thumb': 'thumb', '拇指': 'thumb', '大拇指': 'thumb',
    'index': 'index', '食指': 'index',
    'middle': 'middle', '中指': 'middle'
}
VALID_FINGERS = ['thumb', 'index', 'middle']

def parse_fingers_input(raw: str) -> list:
    raw = raw.strip()
    if raw.lower() in ('none', 'no', 'n', ''):
        return []
    # Normalize the Chinese enumeration comma.
    parts = [p.strip() for p in raw.replace('、', ',').split(',') if p.strip()]
    mapped = []
    for p in parts:
        key = p.lower()
        if key in FINGER_ALIASES:
            norm = FINGER_ALIASES[key]
            if norm not in mapped:
                mapped.append(norm)
        else:
            loguru.warning(f"[FINGER] Unrecognized finger: {p} (allowed: {VALID_FINGERS} or Chinese aliases)")
    return mapped

def annotate_fingers(appeared: list, interactive: bool = True) -> list:
    """Return a contact_finger list: [{hand: 'left', fingers: [...]}, ...]."""
    result = []
    for hand in ['left', 'right']:
        if hand not in appeared:
            continue
        if interactive:
            prompt = (
                f"Enter fingers of the {hand} hand in contact with the object "
                "(comma-separated; supports thumb,index,middle or Chinese aliases; none for no fingers): "
            )
            raw = input(prompt)
            fingers = parse_fingers_input(raw)
        else:
            fingers = []
        result.append({'hand': hand, 'fingers': fingers})
    return result

def annotate_fingers_overwrite(existing: dict | None, interactive: bool = True) -> list:
    """Edit mode (mode2): allow both hands to be re-entered regardless of appeared.
    Existing values can be overwritten.
    existing: previous contact_finger list -> [{'hand': 'left','fingers': [...]}, ...]
    Return the new list.
    Rules:
      - Input none / no / n / empty string => fingers = []
      - Press Enter with an empty string and an existing value for that hand => keep the previous value
    """
    prev_map = {}
    if isinstance(existing, list):
        for item in existing:
            h = item.get('hand')
            if h in ['left','right']:
                prev_map[h] = item.get('fingers', [])
    result = []
    for hand in ['left','right']:
        if interactive:
            prev_display = prev_map.get(hand, [])
            raw = input(
                f"Current {hand} hand value: {prev_display}; enter new fingers "
                "(comma-separated; supports thumb/index/middle or Chinese aliases; "
                "none=empty; press Enter to keep unchanged): "
            ).strip()
            if raw == '' and hand in prev_map:
                fingers = prev_map[hand]
            else:
                fingers = parse_fingers_input(raw)
        else:
            # Non-interactive mode: keep old values, or empty if missing.
            fingers = prev_map.get(hand, [])
        result.append({'hand': hand, 'fingers': fingers})
    return result



def main():
    parser = argparse.ArgumentParser(description='Generate sequence for DMT-align')
    parser.add_argument('--seq-path', type=str, required=True, 
                        help='Path to the sequence directory')
    parser.add_argument('--out', '--o', type=str, default='/processed/ho_contact.json',
                        help='Output directory for the generated sequence')
    parser.add_argument('--mode', type=str, default='1', choices=['1','2','legacy','fingers'],
                        help='Mode: 1/legacy=contact annotation only by marking non-contact ranges; 2/fingers=mark contact ranges and annotate contact fingers for each segment')
    parser.add_argument('--no-interactive', action='store_true', help='For tests: skip all prompts and leave finger results empty')
    
    args = parser.parse_args()
    out_file = f"{args.seq_path}{args.out}"
    img_dir = os.path.join(args.seq_path, 'build/image')
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    frames_cnt = len(img_files)
    
    mode = args.mode
    interactive = not args.no_interactive

    if mode in ('1','legacy'):
        # Original contact annotation flow.
        appeared = []
        if interactive:
            left_appear = input("Did left hand appear? (y/n): ").strip().lower()
            if left_appear == 'y':
                appeared.append('left')
            right_appear = input("Did right hand appear? (y/n): ").strip().lower()
            if right_appear == 'y':
                appeared.append('right')
        else:
            appeared = ['left','right']

        r_contacts = [True for _ in range(frames_cnt)]
        l_contacts = [True for _ in range(frames_cnt)]
        if interactive:
            while True:
                s = input("right-hand NON-contact ranges (e.g., 156-160, 170-175) or 'done' to finish: ")
                if s.lower() == 'done':
                    break
                idx_set = parse_ranges(s, img_files)
                for i in idx_set:
                    r_contacts[i] = False
            while True:
                s = input("left-hand NON-contact ranges (e.g., 156-160, 170-175) or 'done' to finish: ")
                if s.lower() == 'done':
                    break
                idx_set = parse_ranges(s, img_files)
                for i in idx_set:
                    l_contacts[i] = False
        loguru.info(f"Final contact ranges RIGHT={r_contacts} LEFT={l_contacts}")
        contact_list = []
        for i, (cr, cl) in tqdm(enumerate(zip(r_contacts, l_contacts)), total=frames_cnt):
            frame_num = int(os.path.splitext(img_files[i])[0])
            contact_list.append({'frame': frame_num,'r_contact': cr,'l_contact': cl})
        contact_dict = {
            'frames_cnt': frames_cnt,
            'appeared': appeared,
            'contacts': contact_list,
        }
        with open(out_file, 'w') as f:
            json.dump(contact_dict, f, indent=2)
        loguru.info(f"Contact data saved to {out_file}")
        return

    # Mode 2: follow the mode-1 interaction pattern, but mark contact ranges
    # and annotate contact fingers for each segment.
    appeared = []
    if interactive:
        left_appear = input("Did left hand appear? (y/n): ").strip().lower()
        if left_appear == 'y':
            appeared.append('left')
        right_appear = input("Did right hand appear? (y/n): ").strip().lower()
        if right_appear == 'y':
            appeared.append('right')
    else:
        appeared = ['left', 'right']

    l_contacts = [False for _ in range(frames_cnt)]
    r_contacts = [False for _ in range(frames_cnt)]
    l_fingers_per_frame = [set() for _ in range(frames_cnt)]
    r_fingers_per_frame = [set() for _ in range(frames_cnt)]

    contact_segments = {
        'left': [],
        'right': [],
    }

    hand_to_contacts = {
        'left': l_contacts,
        'right': r_contacts,
    }
    hand_to_fingers = {
        'left': l_fingers_per_frame,
        'right': r_fingers_per_frame,
    }

    if interactive:
        for hand in ['left', 'right']:
            if hand not in appeared:
                continue
            while True:
                s = input(
                    f"{hand}-hand CONTACT ranges (e.g., 156-160, 170-175) or 'done' to finish: "
                ).strip()
                if s.lower() == 'done':
                    break

                idx_set = parse_ranges(s, img_files)
                if not idx_set:
                    loguru.warning(f"[{hand}] No valid frame range matched: {s}")
                    continue

                segs = idx_set_to_segments(idx_set)
                for seg_start_idx, seg_end_idx in segs:
                    start_frame = int(os.path.splitext(img_files[seg_start_idx])[0])
                    end_frame = int(os.path.splitext(img_files[seg_end_idx])[0])
                    finger_raw = input(
                        f"[{hand}] Contact fingers for CONTACT segment {start_frame}-{end_frame} "
                        "(thumb,index,middle or Chinese aliases; none=empty): "
                    ).strip()
                    fingers = parse_fingers_input(finger_raw)

                    contact_segments[hand].append({
                        'start': start_frame,
                        'end': end_frame,
                        'fingers': fingers,
                    })

                    for idx in range(seg_start_idx, seg_end_idx + 1):
                        hand_to_contacts[hand][idx] = True
                        hand_to_fingers[hand][idx].update(fingers)

    contact_list = []
    for i in range(frames_cnt):
        frame_num = int(os.path.splitext(img_files[i])[0])
        contact_list.append({
            'frame': frame_num,
            'r_contact': r_contacts[i],
            'l_contact': l_contacts[i],
            'r_fingers': sorted(list(r_fingers_per_frame[i])),
            'l_fingers': sorted(list(l_fingers_per_frame[i])),
        })

    contact_dict = {
        'frames_cnt': frames_cnt,
        'appeared': appeared,
        'contacts': contact_list,
        'contact_segments': contact_segments,
    }

    with open(out_file, 'w') as f:
        json.dump(contact_dict, f, indent=2)
    loguru.info(f"Mode-2 contact+finger annotation saved to {out_file}")

    left_contact_frames = sum(1 for v in l_contacts if v)
    right_contact_frames = sum(1 for v in r_contacts if v)
    left_segments = len(contact_segments.get('left', []))
    right_segments = len(contact_segments.get('right', []))
    loguru.info(
        f"Mode-2 summary | left: {left_contact_frames} contact frames, {left_segments} segments | "
        f"right: {right_contact_frames} contact frames, {right_segments} segments"
    )

if __name__ == "__main__":
    main()
