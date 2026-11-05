"""CIGVis/VisPy physical-coordinate view of memory-mapped sand NPY models."""
from pathlib import Path
import json

import numpy as np
from colorbar_settings import sand_color_style, validate_sand_colors


def plane_geometry(plane):
    n, m = plane['shape']
    ids = np.arange(n * m, dtype='u4').reshape(n, m)
    valid = plane['valid']
    cells = valid[:-1, :-1] & valid[1:, :-1] & valid[1:, 1:] & valid[:-1, 1:]
    a, b, c, d = [array[cells] for array in (ids[:-1, :-1], ids[1:, :-1], ids[1:, 1:], ids[:-1, 1:])]
    faces = np.concatenate((np.stack((a, b, c), axis=1), np.stack((a, c, d), axis=1)))
    return plane['vertices'], faces.astype('u4'), plane['values']


def depth_axes(ranges):
    from vispy.scene.visuals import Line, Text
    x0, x1 = ranges['x']; y0, y1 = ranges['y']; z0, z1 = ranges['z']
    dx, dy, dz = x1-x0, y1-y0, z1-z0
    segments = [[x0,y0,z1], [x1,y0,z1], [x0,y0,z1], [x0,y1,z1], [x0,y0,z1], [x0,y0,z0]]
    labels, positions = [], []
    for value in np.linspace(x0, x1, 4):
        segments.extend([[value,y0,z1], [value,y0-.018*dy,z1]])
        positions.append([value,y0-.05*dy,z1]); labels.append(f'{value:.1f}')
    for value in np.linspace(y0, y1, 4):
        segments.extend([[x0,value,z1], [x0-.018*dx,value,z1]])
        positions.append([x0-.05*dx,value,z1]); labels.append(f'{value:.1f}')
    for value in np.linspace(z1, z0, 4):
        segments.extend([[x0,y0,value], [x0-.018*dx,y0,value]])
        positions.append([x0-.075*dx,y0,value]); labels.append(f'{-value*1000:.0f}')
    # A light bounding box establishes the saved volume's full physical extent.
    for x in (x0,x1):
        for y in (y0,y1):
            segments.extend([[x,y,z0], [x,y,z1]])
    for z in (z0,z1):
        segments.extend([[x0,y0,z],[x1,y0,z],[x1,y0,z],[x1,y1,z],
                         [x1,y1,z],[x0,y1,z],[x0,y1,z],[x0,y0,z]])
    return [Line(pos=np.asarray(segments,dtype='f4'), connect='segments', color='#52657a', width=.8),
        Text(text=labels,pos=np.asarray(positions,dtype='f4'), color='#344154',font_size=8),
        Text(text=['X / km','Y / km','Depth / m (down)'],
             pos=np.asarray([[(x0+x1)/2,y0-.12*dy,z1],[x0-.12*dx,(y0+y1)/2,z1],
                             [x0-.12*dx,y0,(z0+z1)/2]],dtype='f4'), color='#344154',font_size=9)]


class SavedSandScene:
    def __init__(self, model, body, indices, sand_colors=None, vertical_scale=5., opacity=.3,
                 threshold_percent=0., size=(1320, 900)):
        import cigvis
        from cigvis.vispynodes import VisCanvas, Colorbar
        from vispy import app, scene
        from vispy.scene.visuals import Mesh
        from matplotlib.cm import ScalarMappable
        app.use_app('pyside6')
        cigvis.set_order(True); cigvis.set_axis_reversed(False,False,False)
        self.model, self.body = model, body
        self.indices = list(indices)
        self.opacity = opacity
        self.threshold_percent = threshold_percent
        self.vertical_scale = vertical_scale
        self.sand_colors = validate_sand_colors(sand_colors)
        self.style = {}
        self._update_style()
        self.ranges = {'x': (model.axes[0][[0,-1]]/1000).tolist(),
                       'y': (model.axes[1][[0,-1]]/1000).tolist(),
                       'z': [-float(model.axes[2][-1])/1000, -float(model.axes[2][0])/1000]}
        self.plane_nodes = [None]*3; self.plane_data = [None]*3
        self.plane_visible = [True]*3
        self.body_node = None
        nodes = depth_axes(self.ranges)
        if len(body['faces']):
            self.body_node = Mesh(vertices=body['vertices'],faces=body['faces'],
                vertex_colors=self.colors(body['values'], opacity),shading=None)
            self.body_node.set_gl_state('translucent', depth_test=True, depth_mask=False, cull_face=False)
            nodes.append(self.body_node)
        for i,index in enumerate(indices):
            data = plane_geometry(model.plane(i,index))
            self.plane_data[i] = data
            if len(data[1]):
                mesh = Mesh(vertices=data[0],faces=data[1],vertex_colors=self.colors(data[2]),shading=None)
                mesh.set_gl_state('opaque',depth_test=True,cull_face=False,
                                  polygon_offset_fill=True,polygon_offset=(1.,1.))
                self.plane_nodes[i] = mesh; nodes.append(mesh)
        style = self.style
        class SandColorbar(Colorbar):
            def get_ScalarMappable(self):
                return ScalarMappable(norm=style['norm'],cmap=style['cmap']), {
                    'ticks':style['ticks'],'labels':[f'{v:g}' for v in style['ticks']]}
        self.colorbar = SandColorbar(cmap=style['cmap'],clim=[style['norm'].vmin,style['norm'].vmax],
            discrete=True,disc_ticks=[style['ticks']],label_str='Sand volume fraction / %')
        self.colorbar.update_params(cbar_size=(100,620)); nodes.append(self.colorbar)
        class DepthClearingCanvas(VisCanvas):
            def _draw_scene(self,bgcolor=None):
                self.context.set_state(depth_mask=True)
                return super()._draw_scene(bgcolor)
        self.canvas = DepthClearingCanvas(visual_nodes=nodes,size=size,bgcolor='white',
            axis_scales=(1.,1.,vertical_scale),azimuth=-55,elevation=28,fov=35,
            zoom_factor=.86,keys=None,title='Saved 3D sand fraction | depth down')
        self.fit()
        scene.Text('Saved NPY sand model | XY / km | depth down',parent=self.canvas.scene,
                   pos=(20,22),color='#1e3449',font_size=12,anchor_x='left',anchor_y='top')

    def _update_style(self):
        cmap,norm,ticks = sand_color_style(self.sand_colors,percent=True)
        self.style.update(cmap=cmap,norm=norm,ticks=ticks)

    def colors(self,values,alpha=1.):
        rgba = np.asarray(self.style['cmap'](self.style['norm'](values*100)),dtype='f4')
        rgba[:,3] = alpha
        return rgba

    def fit(self):
        view = self.canvas.view[0]
        view.camera.set_range(x=self.ranges['x'],y=self.ranges['y'],z=self.ranges['z'],margin=.16)
        view.camera._flip_factors = [1.,1.,self.vertical_scale]
        view.camera._update_transform(); self.canvas.update()

    def set_vertical_scale(self,value):
        if not np.isfinite(value) or value <= 0:
            raise ValueError('垂向显示夸张须为正数。')
        self.vertical_scale = value
        self.canvas.update_axis_scales((1.,1.,value)); self.fit()

    def update_planes(self,indices,visible=None):
        from vispy.scene.visuals import Mesh
        self.indices = list(indices)
        self.plane_visible = list(visible) if visible is not None else self.plane_visible
        for i,index in enumerate(indices):
            data = plane_geometry(self.model.plane(i,index)); self.plane_data[i] = data
            node = self.plane_nodes[i]
            if len(data[1]):
                if node is None:
                    node = Mesh(parent=self.canvas.view[0].scene,shading=None)
                    node.set_gl_state('opaque',depth_test=True,cull_face=False,
                                      polygon_offset_fill=True,polygon_offset=(1.,1.))
                    self.plane_nodes[i] = node
                node.set_data(vertices=data[0],faces=data[1],vertex_colors=self.colors(data[2]))
                node.visible = self.plane_visible[i]
            elif node is not None:
                node.visible = False
        self.canvas.update()

    def recolor(self,config=None,opacity=None):
        if config is not None:
            self.sand_colors = validate_sand_colors(config); self._update_style()
            self.colorbar.update_params(cmap=self.style['cmap'],
                clim=[self.style['norm'].vmin,self.style['norm'].vmax])
        if opacity is not None:
            self.opacity = opacity
        if self.body_node is not None:
            self.body_node.set_data(vertices=self.body['vertices'],faces=self.body['faces'],
                                    vertex_colors=self.colors(self.body['values'],self.opacity))
        for node,data in zip(self.plane_nodes,self.plane_data):
            if node is not None and len(data[1]):
                node.set_data(vertices=data[0],faces=data[1],vertex_colors=self.colors(data[2]))
        self.canvas.update()

    def set_body_visible(self,visible):
        if self.body_node is not None:
            self.body_node.visible = visible
        self.canvas.update()

    def metadata(self):
        camera = self.canvas.view[0].camera
        return {'source_sand_npy':str(self.model.path),'source_array_order':'X,Y,depth',
            'source_grid_shape_xyz':list(self.model.shape),'body_display_sample_shape':self.body['sample_shape'],
            'body_triangles':int(len(self.body['faces'])),'threshold_percent':self.threshold_percent,
            'surface_opacity':self.opacity,'display_vertical_exaggeration':self.vertical_scale,
            'sand_color_settings':self.sand_colors,'slice_indices_xyz':self.indices,
            'slice_coordinates_m':[float(axis[i]) for axis,i in zip(self.model.axes,self.indices)],
            'slice_visible_xyz':self.plane_visible,
            'body_visible':bool(self.body_node is not None and self.body_node.visible),
            'coordinates':'Original XY kilometres; world Z=-depth/1000; depth positive down',
            'camera':{'azimuth':float(camera.azimuth),'elevation':float(camera.elevation)},
            'source_arrays_modified':False,'renderer':'CIGVis VisCanvas / VisPy / PySide6',
            'display_sampling':'Surface and large planes may be sampled for display; source model stays unchanged.'}

    def save(self,output):
        from vispy import app,io
        output = Path(output); output.parent.mkdir(parents=True,exist_ok=True)
        for _ in range(4):
            app.process_events()
        pixels = self.canvas.render(alpha=False)
        colored = pixels[50:-30,:-140,:3].astype('i2')
        count = int(np.sum(colored.max(axis=2)-colored.min(axis=2)>35))
        has_geometry = (self.body_node is not None and self.body_node.visible) or any(
            node is not None and node.visible for node in self.plane_nodes)
        if not has_geometry:
            raise ValueError('当前没有可见的含砂率图形，请启用切片或降低体表面阈值。')
        metadata = self.metadata(); metadata['visible_colored_pixels'] = count
        center = [float(np.mean(self.ranges[a])) for a in ('x','y')]
        z0,z1 = self.ranges['z']
        points = np.array([[*center,z0+.2*(z1-z0),1],[*center,z0+.8*(z1-z0),1]])
        projected = self.canvas.view[0].scene.node_transform(self.canvas.scene).map(points)
        pixel = projected[:,:2]/projected[:,3,None]
        metadata['deeper_minus_shallower_screen_y_px'] = float(pixel[0,1]-pixel[1,1])
        io.write_png(str(output),pixels)
        output.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        return output

    def close(self):
        self.canvas.close()
