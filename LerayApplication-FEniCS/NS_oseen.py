from dolfin import *
from odeint_BDF2 import * 
## Mesh
nx=100
ny=40
alpha_0 = 1.5
alpha_1 = -2
alpha_2 = 0.5
## Choose partitioner (SCOTCH, ParMETIS)
#parameters["mesh_partitioner"] = "SCOTCH"

## Structured mesh======================================================
## TODO: setup interface with gmsh 
mesh = RectangleMesh(Point(0.0, 0.0), Point(10., 1.0), nx, ny,"right")
with XDMFFile(MPI.comm_world, "output/mesh.xdmf") as xdmf_handler:
    xdmf_handler.write(mesh)
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
Top = CompiledSubDomain("x[1]==side && on_boundary",side=1.) 
Top.mark(boundaries,1)
Bottom = CompiledSubDomain("x[1]==side && on_boundary",side=0.) 
Bottom.mark(boundaries,2)
Left = CompiledSubDomain("x[0]==side && on_boundary",side=0.) 
Left.mark(boundaries,3)
Right = CompiledSubDomain("x[0]==side && on_boundary",side=10.) 
Right.mark(boundaries,4)
with XDMFFile(MPI.comm_world, "output/boundary.xdmf") as xdmf_handler:
    xdmf_handler.write(boundaries)

## Define the measure for integration===================================
dx = Measure('dx',domain=mesh)
ds = Measure('dx',domain=mesh, subdomain_data=boundaries)

## Define the function space============================================
Vel = VectorElement("Lagrange", mesh.ufl_cell(), 2) #P2 
P = FiniteElement("Lagrange", mesh.ufl_cell(), 1) #P1
element = MixedElement([Vel, P])
M_ns = FunctionSpace(mesh, element)

dm_ns  = TrialFunction(M_ns)
ddm_ns = TestFunction(M_ns)
m_ns   = Function(M_ns)
(v, p) = split(m_ns)

## tuple Trial Function of velocity and pressure = split(m)
(du,dp) = split(dm_ns)
# (du_conv,_) = split(dm_ns)
# # tuple Test Function of velocity and pressure =split(dm)
(ddv,ddp) = split(ddm_ns)
# (ddv_conv,_) = split(ddm_ns)

# M_leray = FunctionSpace(mesh,element)
# m_l  = Function(M_leray)
# dm_l = TrialFunction(M_leray)
# ddm_l= TestFunction(M_leray)


## Define the zero initial condition====================================
m0_ns = Function(M_ns)
## for second order time discretization (BDF2)
m00_ns =Function(M_ns)


## Strain Rate
D = lambda v : (grad(v).T + grad(v)) / 2
# # Spin tensor
# Spin = lambda v: (grad(v) - grad(v).T)/2

## Source Term
b = Constant((0.,0.))
## Neumann traction
t = Constant((0.,0.))

## Time integration setup================================================
odeint_BDF2 = odeint_BDF2(dt = 0.01) 
## if dt need to adapted 
# odeint.dt = 0.1
## material parameter=====================================================
rho = Constant(1.)
mu  = Constant(0.001)

## Boundary condition for NS=================================================
inflow_profile = Expression(('40.0*1.5*x[1]*(ymax - x[1]) / pow(ymax, 2)', '0'),ymax=10, degree=2)
# inflow_profile = Expression(('1.0/(h*h)*(x[1]-h)*(x[1]+h)*(-1)','0.0'), h=1., degree=2)

bcs = [DirichletBC(M_ns.sub(0), inflow_profile  , boundaries, 3 )]
bcs.append(DirichletBC(M_ns.sub(0),Constant((0.,0.)),boundaries, 1 )) 
bcs.append(DirichletBC(M_ns.sub(0),Constant((0.,0.)),boundaries, 2 )) 

## Boundary condition for leray=================================================
# bcs_l = [DirichletBC(M_leray.sub(0), Constant((0.,0.))  , boundaries, 1 )]
# bcs_l.append(DirichletBC(M_leray.sub(0),Constant((0.,0.)),boundaries, 2 )) 
# bcs_l.append(DirichletBC(M_leray.sub(0),Constant((0.,0.)),boundaries, 3 )) 
# bcs_l.append(DirichletBC(M_leray.sub(0),Constant((0.,0.)),boundaries, 4 )) 


def f(m0_ns, m00_ns):
    # v, p  = split(m_ns)
    u0,_  = split(m0_ns) 
    u00,_ = split(m00_ns)
    u_star = 2*u0-u00
    f = + inner( ddp         , div(du)                  ) * dx \
        + inner( ddv         , rho * grad(du) * u_star  ) *dx \
        - inner( div(ddv)    , dp                        ) * dx \
        + inner( D(ddv)      , 2*mu * D(du)              ) * dx \
        - inner( ddv         , b                        ) * dx \
        - inner( ddv         , t                        ) * ds(4)
    return f

def g(dmdt, time_instant):
    dvdt, dpdt  = split(dmdt)
    g =  inner( ddv          , rho * dvdt            ) * dx
    return g

## Overall form
F = odeint_BDF2.g_(g, m_ns, m0_ns, m00_ns) + alpha_0 * inner( ddv, rho * du) * dx + f(m0_ns, m00_ns)
# # linearisation
# J = derivative(F, m_ns, dm_ns)
## Define Leray Problem now for filtering 


file_handler_press = XDMFFile('output/press.xdmf')
file_handler_vel = XDMFFile('output/vel.xdmf')
# file_handler_vel_bar = XDMFFile('output/vel_bar.xdmf')
# file_handler_press_bar = XDMFFile('output/press_bar.xdmf')
# file_handler_indicator = XDMFFile('output/indicator.xdmf')

## Timeloop===============================================================
for i in range(20):
    # F = odeint_BDF2.g_(g, m_ns, m0_ns, m00_ns) + alpha_0 * inner( ddv, rho * du) * dx + f(m0_ns, m00_ns)
    # set current time
    t = i * odeint_BDF2.dt
    #if i%10:
    print("Processing time instant = %4.3f in step %d " % (t,i), end='\n')
    #print("Processing time instant = {} in step {} ").format(t,i)
    # solve nonlinear problem
    solve(lhs(F) == rhs(F), m_ns, bcs) # PETScSNES, set option.. 
    

#     # extract solution
    v, p = m_ns.split(True) # deepcopy.
    
#     indicator = project(indicator_Q_criteria(v,delta))
#     indicator.rename('a_Q', 'a_Q')
#     # TODO: put parameter for do not save mesh at each time step.
#     file_handler_indicator.write(indicator,float(t))
    
#     # here we need to check indicator what happen.
#     solve(lhs(F_leray) == rhs(F_leray), m_l, bcs_l ) 

    
#     v_bar, p_bar = m_l.split(True) # deepcopy.

    # write output
    v.rename('vel', 'vel')
    p.rename('press', 'press')
#     # TODO: put parameter for do not save mesh at each time step.
    file_handler_press.write(p,float(t))
    file_handler_vel.write(v,  float(t))
    
#     v_bar.rename('vel_bar', 'vel_bar')
#     p_bar.rename('press_bar', 'press_bar')
#     # TODO: put parameter for do not save mesh at each time step.
#     file_handler_press_bar.write(p_bar,float(t))
#     file_handler_vel_bar.write(v_bar,  float(t))
 
    
#     # update solution states for time integration
    m0_ns, m00_ns = odeint_BDF2.update( m_ns, m0_ns, m00_ns )

print('\nDone.')
















