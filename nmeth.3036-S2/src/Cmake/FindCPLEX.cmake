SET(CPLEX_ROOT_DIR "C:/Program Files/IBM/ILOG/CPLEX_Studio1251/" CACHE PATH "CPLEX root directory")#change this if necessary

FIND_PATH(CPLEX_INCLUDE_DIR ilcplex/cplex.h
 PATHS ${CPLEX_ROOT_DIR}/cplex/include
 PATHS "/opt/ilog/cplex91/include"
 HINTS ${CPLEX_ROOT_DIR}/cplex/include
   )
   
FIND_LIBRARY(CPLEX_LIBRARY NAMES cplex1251
     PATHS ${CPLEX_ROOT_DIR}/cplex/lib/x64_windows_vs2010/stat_mda/
     PATHS ${CPLEX_ROOT_DIR}/cplex/lib/x86_windows_vs2008/stat_mda/
     PATHS "/opt/ilog/concert/bin"
     HINTS ${CPLEX_ROOT_DIR}/cplex/lib/x86_windows_vs2008/stat_mda/
   )    

FIND_LIBRARY(ILOCPLEX_LIBRARY NAMES ilocplex
     PATHS ${CPLEX_ROOT_DIR}/cplex/lib/x64_windows_vs2010/stat_mda/
     PATHS ${CPLEX_ROOT_DIR}/cplex/lib/x86_windows_vs2008/stat_mda/
     PATHS "/opt/ilog/cplex91/bin"
     HINTS ${CPLEX_ROOT_DIR}/cplex/lib/x86_windows_vs2008/stat_mda/
   ) 

INCLUDE(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(CPLEX DEFAULT_MSG
       CPLEX_LIBRARY
       ILOCPLEX_LIBRARY
       CPLEX_INCLUDE_DIR       
       )
   
IF(CPLEX_FOUND)
	SET(CPLEX_INCLUDE_DIRS ${CPLEX_INCLUDE_DIR} )
	SET(CPLEX_LIBRARIES ${CPLEX_LIBRARY} ${ILOCPLEX_LIBRARY} )

	IF(LINUX)
    		SET(CPLEX_LIBRARIES "${CPLEX_LIBRARIES};m;pthread")
	ENDIF(LINUX)
ENDIF(CPLEX_FOUND)

MARK_AS_ADVANCED(    CPLEX_LIBRARY ILOCPLEX_LIBRARY CPLEX_INCLUDE_DIR )