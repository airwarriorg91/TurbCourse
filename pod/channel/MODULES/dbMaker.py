"""
The module creates database for a scalar or vectorial quantity.
"""

from pymech import neksuite
from datetime import datetime




def dbMan_sym(info,iff,if3D,infom,iffm,infos,iffs):
    """
    To manage the databases creation.
    """
    
    db   = dbCreator(info,iff,if3D)
    db_m = dbCreator(infom,iffm,if3D)    
    # Consistency check mass matrix/snapshots mesh
    data_ms = db_m['data'][0]  
    if (db['data'][0].nel != data_ms.nel):
        print('N.elements of mass matrix element != N.elements of velocity fields !!!')
        exit(0)    
    # Symmetric fields database
    db_s = dbCreator(infos,iffs,if3D)
    if (db['data'][0].nel != db_s['data'][0].nel):
        print('N.elements of mirrored field != N.elements of original fields !!!')
        exit(0)   

    return db,db_m,db_s,data_ms




def dbMan(info,iff,if3D,infom,iffm):
    """
    To manage the databases creation.
    """
    
    db   = dbCreator(info,iff,if3D)
    db_m = dbCreator(infom,iffm,if3D)    
    # Consistency check mass matrix/snapshots mesh
    data_ms = db_m['data'][0]  
    if (db['data'][0].nel != data_ms.nel):
        print('N.elements of mass matrix element != N.elements of velocity fields !!!')
        exit(0)    


    return db,db_m,data_ms





def dbCreator(info,iff,if3D):
    """
    The database is created for the solution field or the mass matrix.
    """
    if (iff=='field'):
        db = dbCreator_v(info, if3D)
    elif (iff=='mass'):
        db = dbCreator_m(info)
        
    return db





def dbCreator_v(info, if3D):
    """
    The database is created for the solution field (all the snapshots).
    """
    path_    = info['dataPath']
    caseName_= info['caseName']
    start_   = info['startID']
    end_     = info['endID']

    pre_=caseName_+'0.f'
    nSnap=0
    ll =[]
    
    #------------------------------------------------------#
    # Skips reading of variables other than QOI            #
    # Helps in reducing reducing computational time        #
    #                                                      #
    # Added by Gaurav Gupta (IIST)                         #
    #------------------------------------------------------#

    if if3D:
        vars = ['ux', 'uy', 'uz', 'pressure', 'temperature']
    else:
        vars = ['ux', 'uy', 'pressure', 'temperature']

    if (info['qoiName'] == 'temperature'):   
        vars.remove('temperature')
    elif (info['qoiName'] == 'pressure'):
        vars.remove('pressure') 
    elif (info['qoiName'] == 'uvel'):
        vars.remove('uvel')
    elif (info['qoiName'] == 'vvel'):
        vars.remove('vvel')
    elif (info['qoiName'] == 'wvel'):
        vars.remove('wvel')
    elif (info['qoiName'] == 'velocity'):
        vars.remove('uvel')
        vars.remove('vvel')
        if if3D:
            vars.remove('wvel')   

    for id_ in range(start_,end_+1):
        nSnap+=1
        data = neksuite.readnek(path_+pre_+str(id_).zfill(5),'float32', skip_vars=vars)

        ll.append(data)
    time_=datetime.now()
    time_= time_.strftime("%d/%m/%Y %H:%M:%S")
        
    db={'data':ll,
        'nSnap':nSnap,    
        'creationDate':time_,
        'startFile':pre_+str(start_).zfill(5),
        'endFile':pre_+str(end_+1).zfill(5)
        }

    return db





def dbCreator_m(info):
    """
    The database is created for the mass matrix.
    One snapshots is needed, since they all have the same mesh.
    """
    path_    = info['dataPath']
    caseName_= info['caseName']
    
    pre_=caseName_+'0.f'
    ll =[]

    data = neksuite.readnek(path_+pre_+str(1).zfill(5),'float32')
    
    ll.append(data)
    time_=datetime.now()
    time_= time_.strftime("%d/%m/%Y %H:%M:%S")
        
    db={'data':ll, 
        'creationDate':time_,
        }

    return db
