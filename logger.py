import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from config import LOG_DIR, OUTPUT_DIR


class ExperimentLogger:
    """Протоколирование эксперимента"""

    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = LOG_DIR / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.log = {
            'experiment': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'configuration': {},
            'phases': {},
            'results': {},
            'errors': []
        }

        print(f"📁 Логи будут сохранены в: {self.experiment_dir}")

    def log_config(self, config: Dict):
        self.log['configuration'] = config
        self._save_now()  # Сохраняем сразу

    def log_phase(self, phase_name: str, data: Dict):
        self.log['phases'][phase_name] = {
            'timestamp': datetime.now().isoformat(),
            **data
        }
        self._save_now()  # Сохраняем после каждой фазы

    def log_error(self, error: str, context: Dict = None):
        self.log['errors'].append({
            'timestamp': datetime.now().isoformat(),
            'error': error,
            'context': context or {}
        })
        self._save_now()  # Сохраняем сразу при ошибке

    def _save_now(self):
        """Сохраняет текущее состояние лога"""
        try:
            log_path = self.experiment_dir / "experiment_log.json"

            def convert(obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                elif isinstance(obj, (np.floating,)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
                elif isinstance(obj, Path):
                    return str(obj)
                return str(obj)

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(self.log, f, ensure_ascii=False, indent=2, default=convert)

        except Exception as e:
            print(f"  ⚠ Не удалось сохранить лог: {e}")

    def save(self):
        """Финальное сохранение"""
        self._save_now()
        self._save_text_report()

        print(f"\n✓ Логи сохранены в: {self.experiment_dir}")
        print(f"  - experiment_log.json")
        print(f"  - report.txt")

    def _save_text_report(self):
        """Сохранение текстового отчета"""
        try:
            report_path = self.experiment_dir / "report.txt"

            with open(report_path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("ОТЧЕТ ПО ЭКСПЕРИМЕНТУ\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"Эксперимент: {self.experiment_name}\n")
                f.write(f"Дата: {self.log['timestamp']}\n\n")

                f.write("Конфигурация:\n")
                f.write("-" * 30 + "\n")
                for key, value in self.log['configuration'].items():
                    f.write(f"  {key}: {value}\n")

                f.write("\nФазы эксперимента:\n")
                f.write("-" * 30 + "\n")
                for phase, data in self.log['phases'].items():
                    f.write(f"\n  [{phase}]\n")
                    for key, value in data.items():
                        if key != 'timestamp':
                            # Обрезаем длинные значения
                            val_str = str(value)
                            if len(val_str) > 200:
                                val_str = val_str[:200] + "..."
                            f.write(f"    {key}: {val_str}\n")

                if self.log['errors']:
                    f.write("\nОшибки:\n")
                    f.write("-" * 30 + "\n")
                    for error in self.log['errors']:
                        f.write(f"  [{error['timestamp']}] {error['error']}\n")

                f.write("\n" + "=" * 60 + "\n")
                f.write("Для детальной информации смотрите experiment_log.json\n")

            print(f"  ✓ Текстовый отчет сохранен")

        except Exception as e:
            print(f"  ⚠ Не удалось сохранить отчет: {e}")


# Для совместимости
import numpy as np