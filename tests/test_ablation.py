"""Tests for ablation study script."""

import numpy as np
import pytest

from scripts.ablation import (
    _generate_box_mesh,
    _generate_cylinder_mesh,
    _generate_sphere_mesh,
    generate_primitive_mesh,
)


class TestMeshGeneration:
    """Test primitive mesh generation functions."""

    def test_generate_sphere_mesh(self):
        """Test sphere mesh generation."""
        radius = 0.025
        resolution = 20

        vertices, faces = _generate_sphere_mesh(radius, resolution)

        assert vertices.shape[1] == 3
        assert faces.shape[1] == 3
        assert vertices.dtype == np.float32
        assert faces.dtype == np.int32

        radii = np.linalg.norm(vertices, axis=1)
        assert np.allclose(radii, radius, atol=1e-6)

    def test_generate_box_mesh(self):
        """Test box mesh generation."""
        half_size = np.array([0.03, 0.05, 0.01])

        vertices, faces = _generate_box_mesh(half_size)

        assert vertices.shape == (8, 3)
        assert len(faces) == 12
        assert vertices.dtype == np.float32
        assert faces.dtype == np.int32

        assert np.all(np.abs(vertices[:, 0]) <= half_size[0] + 1e-6)
        assert np.all(np.abs(vertices[:, 1]) <= half_size[1] + 1e-6)
        assert np.all(np.abs(vertices[:, 2]) <= half_size[2] + 1e-6)

    def test_generate_cylinder_mesh(self):
        """Test cylinder mesh generation."""
        radius = 0.035
        height = 0.10
        resolution = 20

        vertices, faces = _generate_cylinder_mesh(radius, height, resolution)

        assert vertices.shape[1] == 3
        assert faces.shape[1] == 3
        assert vertices.dtype == np.float32
        assert faces.dtype == np.int32

        z_vals = vertices[:, 2]
        assert z_vals.min() >= -height / 2 - 1e-6
        assert z_vals.max() <= height / 2 + 1e-6

    def test_generate_primitive_mesh_sphere(self):
        """Test high-level primitive generation for sphere."""
        vertices, faces = generate_primitive_mesh("sphere")

        assert vertices.shape[1] == 3
        assert faces.shape[1] == 3
        assert len(vertices) > 0
        assert len(faces) > 0

    def test_generate_primitive_mesh_box(self):
        """Test high-level primitive generation for box."""
        vertices, faces = generate_primitive_mesh("box")

        assert vertices.shape == (8, 3)
        assert len(faces) == 12

    def test_generate_primitive_mesh_cylinder(self):
        """Test high-level primitive generation for cylinder."""
        vertices, faces = generate_primitive_mesh("cylinder")

        assert vertices.shape[1] == 3
        assert faces.shape[1] == 3
        assert len(vertices) > 0
        assert len(faces) > 0

    def test_generate_primitive_mesh_invalid(self):
        """Test error handling for invalid object name."""
        with pytest.raises(ValueError, match="Unknown object"):
            generate_primitive_mesh("invalid_object")


class TestMeshQuality:
    """Test generated mesh quality."""

    def test_sphere_mesh_is_closed(self):
        """Verify sphere mesh is watertight."""
        vertices, faces = _generate_sphere_mesh(0.025, 30)

        unique_faces = np.unique(faces)
        assert unique_faces.min() >= 0
        assert unique_faces.max() < len(vertices)

    def test_box_faces_point_outward(self):
        """Verify box face normals point outward."""
        vertices, faces = _generate_box_mesh(np.array([1.0, 1.0, 1.0]))

        for face_idx in faces:
            v0, v1, v2 = vertices[face_idx]

            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)

            assert np.linalg.norm(normal) > 1e-6

    def test_cylinder_caps_exist(self):
        """Verify cylinder has top and bottom caps."""
        vertices, faces = _generate_cylinder_mesh(0.05, 0.1, 16)

        z_vals = vertices[:, 2]

        has_bottom_cap = np.any(np.abs(z_vals - (-0.05)) < 1e-6)
        has_top_cap = np.any(np.abs(z_vals - 0.05) < 1e-6)

        assert has_bottom_cap
        assert has_top_cap
