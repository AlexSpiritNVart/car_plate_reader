import time
from datetime import datetime
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

class GroupManager:
    def __init__(self, dist_thresh: float, time_thresh: float):
        self.exception_dict: Dict[str, Dict[str, float]] = {}
        self.groups: List[Dict[str, Any]] = []
        self.dist_thresh = dist_thresh
        self.time_thresh = time_thresh

    def get_distance(self, plate1: str, plate2: str) -> float:
        distance = 0.0
        print('get_distance ',print(type(plate1)), print(type(plate2)))
        if len(plate1) != len(plate2):
            distance += 0.5
        for i in range(min(len(plate1), len(plate2), 8)):
            if plate1[i] == plate2[i]:
                continue
            distance += self.exception_dict.get(plate1[i], {}).get(plate2[i], 1.0)
        return distance

    def add_exception(self, key1: str, key2: str, value: float):
        self.exception_dict.setdefault(key1, {})[key2] = value
        self.exception_dict.setdefault(key2, {})[key1] = value

    def add_number(self, new_number: str, new_image: Any, new_coords: Any):
        print('add_number', print(type(new_number)))
        if "!" in new_number or len(new_number) <= 7:
            return 0
        if not self.groups:
            self.groups.append({
                'numbers': [new_number],
                'time_code': time.time(),
                'image': new_image,
                'coords': [new_coords],
                'last_time_code': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            return 1
        mean_distances = [
            np.mean([self.get_distance(old_number, new_number) for old_number in group['numbers']])
            for group in self.groups
        ]
        min_dist = min(mean_distances)
        idx = np.argmin(mean_distances)
        if min_dist < self.dist_thresh:
            self.groups[idx]['numbers'].append(new_number)
            self.groups[idx]['coords'].append(new_coords)
            self.groups[idx]['last_time_code'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.groups[idx]['image'] = new_image
        else:
            self.groups.append({
                'numbers': [new_number],
                'time_code': time.time(),
                'image': new_image,
                'coords': [new_coords],
                'last_time_code': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

    def get_best_number(self, group: Dict[str, Any]) -> Tuple[str, Any, Any, str]:
        best_plate = ''
        for i in range(8):
            ith_symbols = [number[i] for number in group['numbers'] if len(number) > i]
            if ith_symbols:
                best_plate += max(set(ith_symbols), key=ith_symbols.count)
        if len([1 for i in group['numbers'] if len(i) == 9]) > len(group['numbers']) // 2:
            print('get_best_number', print(type(i)))
            last_symbols = [i[8] for i in group['numbers'] if len(i) == 9]
            if last_symbols:
                best_plate += max(set(last_symbols), key=last_symbols.count)
        return best_plate, group['image'], group['coords'][-1], group['last_time_code']

    def flush_reduce_groups(self, cam_id: int, num_detect_to_send: int) -> List[Tuple[Any, str, Any, int]]:
        tosend = []
        for group in self.groups[:]:
            print("flush_reduce_groups",print(type(group['numbers'])))
            if len(group['numbers']) >= num_detect_to_send:
                best_plate, image, coords, last_time = self.get_best_number(group)
                tosend.append((image, best_plate, coords, cam_id, last_time))
                self.groups.remove(group)
        return tosend

    def reduce_groups(self, cam_id: int, num_detect_to_send: int, send):
        tosend = []
        for group in self.groups[:]:
            if time.time() - group['time_code'] > self.time_thresh or len(group['numbers']) > 6:
                print("reduce_groups", print(type(group['numbers'])))
                if len(group['numbers']) >= num_detect_to_send:
                    best_plate, image, coords, last_time = self.get_best_number(group)
                    tosend.append((image, best_plate, coords, cam_id, last_time))
                self.groups.remove(group)
        return tosend

    def fill_standard_exceptions(self):
        """Заполнение словаря частых OCR-ошибок (можно расширять!)"""
        self.add_exception('O', '0', 0.1)
        self.add_exception('O', '9', 0.3)
        self.add_exception('O', '6', 0.3)
        self.add_exception('0', '9', 0.3)
        self.add_exception('0', '6', 0.2)
        self.add_exception('6', '9', 0.4)
        self.add_exception('1', 'T', 0.7)
        self.add_exception('1', '7', 0.65)
        self.add_exception('7', 'T', 0.3)
        self.add_exception('3', '8', 0.5)
        self.add_exception('5', '6', 0.5)
        self.add_exception('9', '8', 0.9)
        self.add_exception('8', 'B', 0.3)
        self.add_exception('C', '0', 0.8)
        self.add_exception('C', 'O', 0.8)
        self.add_exception('D', '0', 0.7)
        self.add_exception('D', 'O', 0.7)
