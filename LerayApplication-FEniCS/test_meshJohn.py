import pygmsh
import subprocess
from meshReader  import *
def geometryJohn():
    geom = pygmsh.built_in.Geometry()
    L = 2.2
    W = 0.41
    # we nee the center 
    circle = geom.add_circle(
                    x0=[0.2, 0.2, 0.0], radius=0.05, lcar=0.1, num_sections=4, make_surface=True)
    # then we make external surface
    
    rectangle=geom.add_rectangle(0.0, L, 0.0, W, 0.0, lcar=0.1, holes=[circle.line_loop])
    
    for idx,line in enumerate(rectangle.line_loop.lines):
        points=line.points
        for point in points:
            x = point.x
            print("%s-th line has extrema %s %s %s \t"%(idx,x[0],x[1],x[2]))
    field0 = geom.add_boundary_layer(
                edges_list=[rectangle.line_loop.lines[i] for i in range(0,4)],
                            hfar=0.1,
                            hwall_n=0.01,
                            ratio=1.1,
                            thickness=0.2,
                            anisomax=100.0)
     

    geom.add_background_field([field0])

    geom.add_physical_surface(rectangle.surface)
    
    for i in range(0,4):
       geom.add_physical_line(rectangle.line_loop.lines[i])
    for  idx,line in enumerate(circle.line_loop.lines):
       geom.add_physical_line(line)
    points, cells, _, _, _ = pygmsh.generate_mesh(geom,dim=2,extra_gmsh_arguments=['-format', 'msh2'], geo_filename="John.geo")
    args = [
        "-{}".format(2),
        "John.geo",
        # Don't use the native msh format. It's not very well suited for
        # efficient reading as every cell has to be read individually.
        "-format",
        "msh2",
        "-bin",
        "-o",
        "John.msh"
    ] 

    # https://stackoverflow.com/a/803421/353337
    p = subprocess.Popen(
        ["gmsh"] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    p.communicate()
    return geom

if __name__ == "__main__":
    geom=geometryJohn()
    readMesh("John.msh",file_type="gmsh")
