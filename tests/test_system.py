from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from reconocimiento_facial.cli import main
from reconocimiento_facial.database import UserDatabase
from reconocimiento_facial.dependencies import quiet_model_warnings
from reconocimiento_facial.detectors import FaceDetection
from reconocimiento_facial.extractors import TransFaceBExtractor
from reconocimiento_facial.models import RetinaFaceAssets, TransFaceAssets
from reconocimiento_facial.system import SistemaReconocimientoFacial

LANDMARKS = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)


class FixedExtractor:
    def extract(self, image: np.ndarray) -> np.ndarray:
        if image.shape != (112, 112, 3):
            raise ValueError("Face is not aligned.")
        return np.full(512, 1 / np.sqrt(512), dtype=np.float32)


class OneHotExtractor:
    """Different uniform brightness levels produce orthogonal templates."""

    def extract(self, image: np.ndarray) -> np.ndarray:
        if image.shape != (112, 112, 3):
            raise ValueError("Face is not aligned.")
        template = np.zeros(512, dtype=np.float32)
        template[int(round(image.mean())) % 512] = 1
        return template


class FixedDetector:
    def __init__(self, faces: list[FaceDetection]) -> None:
        self.faces = faces

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        return self.faces


class SystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = UserDatabase(Path(self.temp_dir.name) / "users.csv")
        self.image = np.zeros((120, 120, 3), dtype=np.uint8)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_enrolls_exactly_one_face(self) -> None:
        system = SistemaReconocimientoFacial(
            database=self.database,
            detector=FixedDetector(
                [FaceDetection((10, 10, 50, 50), 0.99, LANDMARKS)]
            ),
            extractor=FixedExtractor(),
        )
        system.inscribir("ana", self.image)
        self.assertEqual(self.database.list_users()[0].user_id, "ana")
        record = self.database.list_users()[0]
        self.assertEqual(record.status, "enrolled")
        expected = FixedExtractor().extract(self.image[:112, :112])
        np.testing.assert_array_equal(record.template, expected)
        with self.assertRaises(ValueError):
            system.inscribir("ana", self.image)
        self.assertEqual(len(self.database.list_users()), 1)
        archived = (self.database.path.parent / record.source_image).resolve()
        np.testing.assert_array_equal(cv2.imread(str(archived)), self.image)
        self.assertEqual(len(list(archived.parent.iterdir())), 1)

    def test_archives_original_bytes_for_auditing(self) -> None:
        original = Path(self.temp_dir.name) / "original.jpeg"
        self.assertTrue(cv2.imwrite(str(original), self.image))
        original_bytes = original.read_bytes()
        system = SistemaReconocimientoFacial(
            database=self.database,
            detector=FixedDetector(
                [FaceDetection((10, 10, 50, 50), 0.99, LANDMARKS)]
            ),
            extractor=FixedExtractor(),
        )
        system.inscribir("ana", original)
        archived = (
            self.database.path.parent
            / self.database.list_users()[0].source_image
        ).resolve()
        original.unlink()
        self.assertEqual(archived.read_bytes(), original_bytes)
        self.assertEqual(
            archived.parent, (self.database.path.parent / "images").resolve()
        )
        self.database.remove_user("ana")
        self.assertTrue(archived.is_file())

    def test_failed_csv_write_removes_archived_image(self) -> None:
        system = SistemaReconocimientoFacial(
            database=self.database,
            detector=FixedDetector(
                [FaceDetection((10, 10, 50, 50), 0.99, LANDMARKS)]
            ),
            extractor=FixedExtractor(),
        )
        with patch.object(
            self.database, "_write", side_effect=OSError("Disco")
        ):
            with self.assertRaises(OSError):
                system.inscribir("ana", self.image)
        self.assertEqual(
            list((self.database.path.parent / "images").iterdir()), []
        )

    def test_users_list_omits_template(self) -> None:
        self.database.add_user(
            "ana",
            "image.jpg",
            template=FixedExtractor().extract(self.image[:112, :112]),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(
                ["--database", str(self.database.path), "users", "list"]
            )
        self.assertEqual(result, 0)
        self.assertEqual(
            output.getvalue().splitlines()[0].split(),
            ["User", "Status", "Image"],
        )
        self.assertEqual(
            output.getvalue().splitlines()[1].split(),
            ["ana", "enrolled", "image.jpg"],
        )

    def test_filters_external_warnings_only_within_context(self) -> None:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with quiet_model_warnings():
                for module in (
                    "torch.cuda.amp.autocast_mode",
                    "torchvision.models._utils",
                    "timm.models.layers",
                    "transface_vit",
                ):
                    warnings.warn_explicit(
                        "Dependencia",
                        UserWarning,
                        "external.py",
                        1,
                        module=module,
                    )
                warnings.warn("Aviso propio", UserWarning)
            warnings.warn_explicit(
                "Fuera del contexto",
                UserWarning,
                "external.py",
                1,
                module="torch",
            )
        self.assertEqual(
            [str(item.message) for item in captured],
            ["Aviso propio", "Fuera del contexto"],
        )

    def test_rejects_zero_or_multiple_faces(self) -> None:
        face = FaceDetection((10, 10, 50, 50), 0.99, LANDMARKS)
        for faces in ([], [face, face]):
            system = SistemaReconocimientoFacial(
                database=self.database, detector=FixedDetector(faces)
            )
            with self.assertRaises(ValueError):
                system.inscribir("ana", self.image)
        self.assertFalse(self.database.path.exists())

    def test_aligns_similarity_transform(self) -> None:
        system = SistemaReconocimientoFacial(detector=FixedDetector([]))
        image = np.random.default_rng(7).integers(
            0, 256, (112, 112, 3), dtype=np.uint8
        )
        enlarged = cv2.resize(
            image, (224, 224), interpolation=cv2.INTER_NEAREST
        )
        aligned = system.alinear(enlarged, np.asarray(LANDMARKS) * 2)
        np.testing.assert_array_equal(aligned, image)
        for points in ((), np.zeros((5, 2)), np.full((5, 2), np.nan)):
            with self.assertRaises(ValueError):
                system.alinear(image, points)

    def test_compares_with_cosine_similarity(self) -> None:
        system = SistemaReconocimientoFacial(
            database=self.database,
            detector=FixedDetector([]),
            extractor=FixedExtractor(),
        )
        a = np.array([3.0, 4.0, 0.0])
        b = np.array([0.0, 0.0, 2.0])
        c = np.array([1.0, 1.0, 1.0])
        self.assertAlmostEqual(system.comparar(a, a), 1.0)
        self.assertAlmostEqual(system.comparar(a, -a), -1.0)
        self.assertAlmostEqual(system.comparar(a, b), 0.0)
        self.assertAlmostEqual(system.comparar(a, c), system.comparar(c, a))
        for invalid in (np.ones(2), np.zeros(3), np.array([np.nan, 1, 1])):
            with self.assertRaises(ValueError):
                system.comparar(a, invalid)

    def test_verifies_and_identifies_templates(self) -> None:
        system = SistemaReconocimientoFacial(
            database=self.database,
            detector=FixedDetector([]),
            extractor=OneHotExtractor(),
        )
        e0, e1, e2 = np.eye(512)[:3]
        system.inscribir_t("ana", 2 * e0)
        system.inscribir_t("luis", e1)
        self.database.add_user("pendiente")
        np.testing.assert_allclose(self.database.list_users()[0].template, e0)
        self.assertTrue(system.verificar_t("ana", e0 + 0.1 * e2))
        self.assertFalse(system.verificar_t("ana", e1 + 0.1 * e2))
        self.assertFalse(system.verificar_t("desconocido", e2))
        self.assertFalse(system.verificar_t("pendiente", e2))
        self.assertEqual(system.identificar_t(e1 + 0.1 * e2), "luis")
        self.assertIsNone(system.identificar_t(e2))
        self.assertEqual(system.identificar_t(e0 + 0.5 * e1), "ana")
        with self.assertRaises(ValueError):
            system.inscribir_t("cero", np.zeros(512))
        with self.assertRaises(ValueError):
            SistemaReconocimientoFacial(
                database=self.database,
                detector=FixedDetector([]),
                extractor=OneHotExtractor(),
                identification_threshold=1.5,
            )

    def test_rejects_exact_database_template_copies_only_at_direct_inputs(
        self,
    ) -> None:
        system = SistemaReconocimientoFacial(
            database=self.database,
            detector=FixedDetector([]),
            extractor=OneHotExtractor(),
        )
        system.inscribir_t("ana", np.eye(512)[0])
        system.inscribir_t("luis", np.eye(512)[1])
        before = self.database.path.read_bytes()
        for record in self.database.list_users():
            for dtype in (np.float32, np.float64):
                copied = np.array(record.template, dtype=dtype).copy()
                with self.subTest(user=record.user_id, dtype=dtype):
                    for claimed in ("ana", "luis", "desconocido"):
                        with self.assertRaisesRegex(
                            ValueError, "Possible template compromise"
                        ):
                            system.verificar_t(claimed, copied)
                    with self.assertRaisesRegex(
                        ValueError, "Possible template compromise"
                    ):
                        system.identificar_t(copied)
                    # A one-epsilon difference makes this a distinct probe.
                    distinct = copied.copy()
                    distinct[2] = np.finfo(dtype).eps
                    self.assertTrue(
                        system.verificar_t(record.user_id, distinct)
                    )
                    self.assertEqual(
                        system.identificar_t(distinct), record.user_id
                    )
        self.assertEqual(before, self.database.path.read_bytes())

    def test_thresholds_are_independent(self) -> None:
        e0, e1 = np.eye(512)[:2]
        probe = e0 + e1  # coseno 0.707 con e0
        system = SistemaReconocimientoFacial(
            database=self.database,
            detector=FixedDetector([]),
            extractor=OneHotExtractor(),
            verification_threshold=0.7,
            identification_threshold=0.71,
        )
        system.inscribir_t("ana", e0)
        self.assertTrue(system.verificar_t("ana", probe))
        self.assertIsNone(system.identificar_t(probe))
        system.verification_threshold, system.identification_threshold = (
            0.71,
            0.7,
        )
        self.assertFalse(system.verificar_t("ana", probe))
        self.assertEqual(system.identificar_t(probe), "ana")

    def test_photo_pipelines_with_and_without_detector(self) -> None:
        system = SistemaReconocimientoFacial(
            database=self.database,
            detector=FixedDetector(
                [FaceDetection((10, 10, 50, 50), 0.99, LANDMARKS)]
            ),
            extractor=OneHotExtractor(),
        )

        def face(value: int, size: int = 120) -> np.ndarray:
            return np.full((size, size, 3), value, np.uint8)

        system.inscribir("ana", face(3))
        self.assertTrue(system.verificar("ana", face(3)))
        self.assertFalse(system.verificar("ana", face(4)))
        self.assertEqual(system.identificar(face(3)), "ana")
        self.assertIsNone(system.identificar(face(4)))

        system.inscribir_sd("luis", face(5, size=200))
        self.assertTrue(system.verificar_sd("luis", face(5, size=112)))
        self.assertFalse(system.verificar_sd("ana", face(5, size=112)))
        self.assertEqual(system.identificar_sd(face(5, size=80)), "luis")
        self.assertIsNone(system.identificar_sd(face(6)))
        for user in self.database.list_users():
            self.assertTrue(
                (self.database.path.parent / user.source_image).is_file()
            )
        with self.assertRaises(ValueError):
            system.verificar_sd("luis", np.zeros((112, 112), np.uint8))

    def test_failed_extraction_does_not_write_csv(self) -> None:
        class BrokenExtractor:
            def extract(self, image: np.ndarray) -> np.ndarray:
                raise RuntimeError("No hay pesos.")

        system = SistemaReconocimientoFacial(
            database=self.database,
            detector=FixedDetector(
                [FaceDetection((10, 10, 50, 50), 0.99, LANDMARKS)]
            ),
            extractor=BrokenExtractor(),
        )
        with self.assertRaises(RuntimeError):
            system.inscribir("ana", self.image)
        self.assertFalse(self.database.path.exists())

    def test_legacy_csv_preserves_pending_users(self) -> None:
        self.database.path.write_text(
            "user_id,status,source_image\nana,pending_template,old.jpg\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "database migrate"):
            self.database.list_users()
        self.assertTrue(self.database.migrate_csv())
        self.assertFalse(self.database.migrate_csv())
        self.database.add_user(
            "bob", template=FixedExtractor().extract(self.image[:112, :112])
        )
        users = self.database.list_users()
        self.assertEqual(users[0].template, ())
        self.assertEqual(users[0].source_image, "old.jpg")
        self.assertEqual(len(users[1].template), 512)
        self.database.remove_user("bob")
        self.assertEqual(self.database.list_users(), users[:1])

    def test_database_rejects_wrong_password_and_tampered_data(self) -> None:
        self.database.add_user("identidad_privada", "ruta_privada/foto.jpg")
        original = self.database.path.read_bytes()
        for plaintext in (
            b"identidad_privada",
            b"ruta_privada",
            b"pending_template",
        ):
            self.assertNotIn(plaintext, original)

        unauthorized = UserDatabase(self.database.path, password="incorrecta")
        with self.assertRaisesRegex(ValueError, "Incorrect password"):
            unauthorized.add_user("intruso")
        self.assertEqual(self.database.path.read_bytes(), original)

        tampered = original[:-1] + bytes([original[-1] ^ 1])
        self.database.path.write_bytes(tampered)
        with self.assertRaisesRegex(ValueError, "damaged or modified"):
            self.database.list_users()
        with self.assertRaisesRegex(ValueError, "damaged or modified"):
            self.database.change_password("otra contraseña")
        self.assertEqual(self.database.path.read_bytes(), tampered)

    def test_cli_changes_password_and_preserves_registered_templates(
        self,
    ) -> None:
        self.database.add_user("ana", "foto.jpg", template=np.eye(512)[0])
        before = self.database.list_users()
        original = self.database.path.read_bytes()
        command = [
            "--database",
            str(self.database.path),
            "database",
            "change-password",
        ]
        new_password = "otra contraseña"
        with patch(
            "reconocimiento_facial.cli.getpass",
            side_effect=[new_password, "distinta"],
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(command), 2)
        self.assertEqual(self.database.path.read_bytes(), original)

        with patch(
            "reconocimiento_facial.cli.getpass",
            side_effect=[new_password, new_password],
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(command), 0)
        with self.assertRaisesRegex(ValueError, "Incorrect password"):
            self.database.list_users()
        with patch.dict("os.environ", {"BIOMETRIC_DB_PASSWORD": new_password}):
            reopened = UserDatabase(self.database.path)
            self.assertEqual(reopened.list_users(), before)
            reopened.add_user("luis")
            self.assertEqual(len(reopened.list_users()), 2)

    def test_database_can_set_password_before_enrolling(self) -> None:
        self.database.change_password("contraseña inicial")
        self.assertTrue(self.database.path.is_file())
        self.assertEqual(self.database.list_users(), [])
        self.database.add_user("ana")
        with self.assertRaisesRegex(ValueError, "Incorrect password"):
            UserDatabase(
                self.database.path, password="incorrecta"
            ).list_users()

    def test_csv_rejects_invalid_templates_without_writing(self) -> None:
        templates = (
            np.zeros(512),
            np.ones(512),
            np.ones(511),
            np.full(512, np.nan),
        )
        for template in templates:
            with self.assertRaises(ValueError):
                self.database.add_user("ana", template=template)
        self.assertFalse(self.database.path.exists())

    def test_rejects_missing_image(self) -> None:
        system = SistemaReconocimientoFacial(
            database=self.database, detector=FixedDetector([])
        )
        with self.assertRaises(FileNotFoundError):
            system.detectar("missing.jpg")

    def test_csv_management(self) -> None:
        self.database.add_user("ana")
        self.assertTrue(self.database.remove_user("ana"))
        self.assertFalse(self.database.remove_user("ana"))

    def test_detection_result_preserves_five_landmarks(self) -> None:
        landmarks = (
            (1.0, 2.0),
            (3.0, 4.0),
            (5.0, 6.0),
            (7.0, 8.0),
            (9.0, 10.0),
        )
        system = SistemaReconocimientoFacial(
            database=self.database,
            detector=FixedDetector(
                [FaceDetection((10, 10, 50, 50), 0.99, landmarks)]
            ),
        )
        self.assertEqual(system.detectar(self.image)[0].landmarks, landmarks)

    def test_detection_cli_writes_annotated_image(self) -> None:
        if (
            not RetinaFaceAssets().weights_path.is_file()
            or not TransFaceAssets().weights_path.is_file()
        ):
            self.skipTest("Pretrained models are missing.")
        image_path = Path("database/enrollment/adele_2016.jpg")
        if not image_path.is_file():
            self.skipTest("Missing the enrollment sample image.")
        output_path = self.database.path.parent / "detection.jpg"
        with contextlib.redirect_stdout(io.StringIO()):
            result = main(
                ["detect", str(image_path), "--output", str(output_path)]
            )
        self.assertEqual(result, 0)
        self.assertIsNotNone(cv2.imread(str(output_path)))

    def test_real_transface_enrollment(self) -> None:
        if (
            not RetinaFaceAssets().weights_path.is_file()
            or not TransFaceAssets().weights_path.is_file()
        ):
            self.skipTest("Pretrained models are missing.")
        if not Path("database/enrollment/adele_2016.jpg").is_file():
            self.skipTest("Missing the enrollment sample image.")
        with contextlib.redirect_stdout(io.StringIO()):
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                result = main(
                    [
                        "--database",
                        str(self.database.path),
                        "enroll",
                        "prueba",
                        "database/enrollment/adele_2016.jpg",
                    ]
                )
        self.assertEqual(result, 0)
        self.assertEqual(errors.getvalue(), "")
        record = self.database.list_users()[0]
        self.assertEqual(record.user_id, "prueba")
        self.assertEqual(record.status, "enrolled")
        self.assertEqual(len(record.template), 512)
        self.assertAlmostEqual(
            float(np.linalg.norm(record.template)), 1.0, places=6
        )

    def test_transface_rgb_scaling_and_l2(self) -> None:
        import torch

        class CaptureModel(torch.nn.Module):
            def forward(
                self, tensor: torch.Tensor
            ) -> tuple[torch.Tensor, None, None]:
                self.input = tensor
                self.grad_enabled = torch.is_grad_enabled()
                return torch.arange(1, 513).float().reshape(1, 512), None, None

        model = CaptureModel()
        extractor = TransFaceBExtractor()
        extractor._model = model
        extractor._device = torch.device("cpu")
        face = np.zeros((112, 112, 3), dtype=np.uint8)
        face[:, :, 2] = 255
        template = extractor.extract(face)
        np.testing.assert_array_equal(
            model.input[0, :, 0, 0].numpy(), [1, -1, -1]
        )
        self.assertFalse(model.grad_enabled)
        self.assertEqual(template.shape, (512,))
        self.assertEqual(template.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(template)), 1.0, places=6)
        for invalid in (face[:100], face.astype(np.float32)):
            with self.assertRaises(ValueError):
                extractor.extract(invalid)
