import logging
from typing import TYPE_CHECKING

from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
)
from vtkmodules.vtkFiltersSources import vtkCubeSource

if TYPE_CHECKING:
    from app.entry import PCDStreamer

from app.utils.logger import setup_logger
logger = setup_logger(__name__)

def update_bounding_box(self: 'PCDStreamer'):
    """Update the bounding box actor."""
    # Create a cube source with the bounding box dimensions
    cube = vtkCubeSource()
    cube.SetBounds(
        self.bbox_params['xmin'], self.bbox_params['xmax'],
        self.bbox_params['ymin'], self.bbox_params['ymax'],
        self.bbox_params['zmin'], self.bbox_params['zmax']
    )

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(cube.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1, 0, 0)  # Red color
    actor.GetProperty().SetRepresentationToWireframe()

    # Remove old bounding box actor if it exists
    if hasattr(self, 'bbox_actor'):
        self.renderer.RemoveActor(self.bbox_actor)

    # Add new bounding box actor
    self.bbox_actor = actor
    self.renderer.AddActor(self.bbox_actor)
    self.vtk_widget.GetRenderWindow().Render()

def on_bbox_slider_changed(self: 'PCDStreamer', value, param):
    self.bbox_params[param] = value / 100.0  # Assuming slider range is scaled
    # Update the corresponding spin box
    self.bbox_edits[param].setValue(self.bbox_params[param])
    update_bounding_box(self)

def on_bbox_edit_changed(self: 'PCDStreamer', value, param):
    self.bbox_params[param] = value
    # Update the corresponding slider
    self.bbox_sliders[param].setValue(int(self.bbox_params[param] * 100))
    update_bounding_box(self)